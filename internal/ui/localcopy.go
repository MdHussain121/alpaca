package ui

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	"github.com/MdHussain121/alpaca/internal/gguf"
	"github.com/MdHussain121/alpaca/internal/hw"
)

const localCopyMarginBytes uint64 = 1 << 30 // keep at least 1 GiB free on target drive

type copyProgressMsg struct {
	doneBytes     int64
	totalBytes    int64
	speedMBps     float64
	eta           time.Duration
	destPath      string
	done          bool
	alreadyExists bool
	err           error
}

type copyFileItem struct {
	src  string
	dst  string
	size int64
}

func estimateModelLoad(path string, totalBytes uint64) (string, time.Duration) {
	if path == "" {
		return "unknown", 0
	}
	kind, _ := hw.DriveKind(path)
	if totalBytes == 0 {
		return kind, 0
	}
	mbps := hw.EstimatedReadMBps(path)
	if mbps <= 0 {
		mbps = 24.0
	}
	sec := float64(totalBytes) / (mbps * 1024 * 1024)
	return kind, time.Duration(sec * float64(time.Second))
}

func healthTimeoutForETA(eta time.Duration) time.Duration {
	const minTimeout = 10 * time.Minute
	const maxTimeout = 60 * time.Minute
	if eta <= 0 {
		return minTimeout
	}
	d := eta * 2
	if d < minTimeout {
		return minTimeout
	}
	if d > maxTimeout {
		return maxTimeout
	}
	return d
}

func fmtETA(d time.Duration) string {
	if d < time.Second {
		return "<1s"
	}
	d = d.Round(time.Second)
	if d < time.Minute {
		return fmt.Sprintf("~%ds", int(d.Seconds()))
	}
	mins := int(d / time.Minute)
	secs := int((d % time.Minute) / time.Second)
	if mins >= 10 || secs == 0 {
		return fmt.Sprintf("~%dm", mins)
	}
	return fmt.Sprintf("~%dm%02ds", mins, secs)
}

func waitCopyProgressCmd(ch <-chan copyProgressMsg) tea.Cmd {
	return func() tea.Msg {
		msg, ok := <-ch
		if !ok {
			return nil
		}
		return msg
	}
}

func copyModelToLocal(
	ctx context.Context,
	srcModelPath, mmprojPath, destDir string,
	diskFreeFn func(string) (uint64, uint64, error),
	ch chan copyProgressMsg,
) {
	defer close(ch)

	sendFinal := func(msg copyProgressMsg) {
		for {
			select {
			case <-ch:
			default:
				select {
				case ch <- msg:
				default:
				}
				return
			}
		}
	}

	if srcModelPath == "" || destDir == "" {
		sendFinal(copyProgressMsg{done: true, err: errors.New("no model selected")})
		return
	}
	if strings.EqualFold(filepath.Clean(filepath.Dir(srcModelPath)), filepath.Clean(destDir)) {
		sendFinal(copyProgressMsg{done: true, err: errors.New("model is already in local folder")})
		return
	}
	if diskFreeFn == nil {
		diskFreeFn = hw.DiskFreeBytes
	}

	var items []copyFileItem
	var totalBytes int64
	allMatch := true
	for _, sp := range gguf.ShardPaths(srcModelPath) {
		st, err := os.Stat(sp)
		if err != nil {
			sendFinal(copyProgressMsg{done: true, err: err})
			return
		}
		dst := filepath.Join(destDir, filepath.Base(sp))
		if dstSt, derr := os.Stat(dst); derr != nil || dstSt.Size() != st.Size() {
			allMatch = false
		}
		items = append(items, copyFileItem{src: sp, dst: dst, size: st.Size()})
		totalBytes += st.Size()
	}
	if mmprojPath != "" && !strings.EqualFold(filepath.Clean(filepath.Dir(mmprojPath)), filepath.Clean(destDir)) {
		if st, err := os.Stat(mmprojPath); err == nil && !st.IsDir() {
			dst := filepath.Join(destDir, filepath.Base(mmprojPath))
			if dstSt, derr := os.Stat(dst); derr != nil || dstSt.Size() != st.Size() {
				allMatch = false
			}
			items = append(items, copyFileItem{src: mmprojPath, dst: dst, size: st.Size()})
			totalBytes += st.Size()
		}
	}

	destFirst := items[0].dst
	if allMatch {
		sendFinal(copyProgressMsg{
			done:          true,
			alreadyExists: true,
			destPath:      destFirst,
			doneBytes:     totalBytes,
			totalBytes:    totalBytes,
		})
		return
	}

	if err := os.MkdirAll(destDir, 0o755); err != nil {
		if errors.Is(err, os.ErrPermission) {
			sendFinal(copyProgressMsg{done: true, err: errors.New("destination folder is read-only; copy blocked")})
			return
		}
		sendFinal(copyProgressMsg{done: true, err: err})
		return
	}
	if !hw.DirWritable(destDir) {
		sendFinal(copyProgressMsg{done: true, err: errors.New("destination folder is read-only; write permission denied")})
		return
	}
	if strings.EqualFold(hw.VolumeFilesystem(destDir), "FAT32") {
		for _, it := range items {
			if it.size > hw.FAT32MaxFileBytes {
				sendFinal(copyProgressMsg{
					done: true,
					err: fmt.Errorf("FAT32 max file size is 4 GB (%s is %.1f GB); format as exFAT/NTFS or split shards",
						filepath.Base(it.src), float64(it.size)/(1<<30)),
				})
				return
			}
		}
	}

	avail, _, err := diskFreeFn(destDir)
	if err == nil {
		need := uint64(totalBytes)
		if avail < need || avail-need < localCopyMarginBytes {
			sendFinal(copyProgressMsg{
				done: true,
				err: fmt.Errorf("insufficient local disk space (need %.1f GB + 1 GB margin, %.1f GB free)",
					float64(need)/(1<<30), float64(avail)/(1<<30)),
			})
			return
		}
	}

	started := time.Now()
	lastEmit := started
	var doneBytes int64
	buf := make([]byte, 4*1024*1024)

	for _, it := range items {
		if err := copySingleFileAtomic(ctx, it, buf, totalBytes, &doneBytes, started, &lastEmit, ch); err != nil {
			sendFinal(copyProgressMsg{done: true, doneBytes: doneBytes, totalBytes: totalBytes, err: err})
			return
		}
	}

	elapsed := time.Since(started).Seconds()
	speed := 0.0
	if elapsed > 0 {
		speed = (float64(doneBytes) / (1024 * 1024)) / elapsed
	}
	sendFinal(copyProgressMsg{
		done:       true,
		destPath:   destFirst,
		doneBytes:  doneBytes,
		totalBytes: totalBytes,
		speedMBps:  speed,
	})
}

func copySingleFileAtomic(
	ctx context.Context,
	it copyFileItem,
	buf []byte,
	totalBytes int64,
	doneBytes *int64,
	started time.Time,
	lastEmit *time.Time,
	ch chan<- copyProgressMsg,
) error {
	src, err := os.Open(it.src)
	if err != nil {
		return err
	}
	defer src.Close()

	partPath := it.dst + ".part"
	_ = os.Remove(partPath)
	dst, err := os.OpenFile(partPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	cleanup := true
	defer func() {
		dst.Close()
		if cleanup {
			_ = os.Remove(partPath)
		}
	}()

	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		n, rerr := src.Read(buf)
		if n > 0 {
			written := 0
			for written < n {
				wn, werr := dst.Write(buf[written:n])
				written += wn
				if werr != nil {
					return werr
				}
			}
			*doneBytes += int64(n)
			now := time.Now()
			if now.Sub(*lastEmit) >= 100*time.Millisecond {
				*lastEmit = now
				elapsed := now.Sub(started).Seconds()
				speed := 0.0
				var eta time.Duration
				if elapsed > 0 && *doneBytes > 0 {
					bps := float64(*doneBytes) / elapsed
					speed = bps / (1024 * 1024)
					if rem := totalBytes - *doneBytes; rem > 0 && bps > 0 {
						eta = time.Duration(float64(rem) / bps * float64(time.Second))
					}
				}
				select {
				case ch <- copyProgressMsg{
					doneBytes:  *doneBytes,
					totalBytes: totalBytes,
					speedMBps:  speed,
					eta:        eta,
				}:
				default:
				}
			}
		}
		if rerr == io.EOF {
			break
		}
		if rerr != nil {
			return rerr
		}
	}

	if err := dst.Close(); err != nil {
		return err
	}
	if err := os.Rename(partPath, it.dst); err != nil {
		return err
	}
	cleanup = false
	return nil
}
