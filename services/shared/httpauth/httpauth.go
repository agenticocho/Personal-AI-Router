// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package httpauth applies credentials held in private local files to HTTP
// requests without placing their contents in configuration, argv, or errors.
package httpauth

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
)

const MaxBearerTokenFileBytes = 16 * 1024

var windowsEnvRE = regexp.MustCompile(`%([^%]+)%`)

// ApplyBearerTokenFile reads tokenFile for this request and replaces the
// Authorization header with its bearer value. An empty path is a no-op.
func ApplyBearerTokenFile(req *http.Request, tokenFile string) error {
	tokenFile = strings.TrimSpace(tokenFile)
	if tokenFile == "" {
		return nil
	}

	file, err := os.Open(expandPath(tokenFile))
	if err != nil {
		return errors.New("bearer token file is unavailable")
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return errors.New("bearer token file metadata is unavailable")
	}
	if !info.Mode().IsRegular() {
		return errors.New("bearer token file is not a regular file")
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return errors.New(
			"bearer token file must not be accessible by group or other",
		)
	}

	data, err := io.ReadAll(
		io.LimitReader(file, MaxBearerTokenFileBytes+1),
	)
	if err != nil {
		return errors.New("bearer token file could not be read")
	}
	if len(data) > MaxBearerTokenFileBytes {
		return fmt.Errorf(
			"bearer token file exceeds %d bytes",
			MaxBearerTokenFileBytes,
		)
	}
	if bytes.IndexByte(data, 0) >= 0 {
		return errors.New("bearer token file contains a NUL byte")
	}

	switch {
	case bytes.HasSuffix(data, []byte("\r\n")):
		data = data[:len(data)-2]
	case bytes.HasSuffix(data, []byte("\n")):
		data = data[:len(data)-1]
	}
	if bytes.ContainsAny(data, "\r\n") {
		return errors.New("bearer token file contains multiple lines")
	}

	token := strings.TrimSpace(string(data))
	if token == "" {
		return errors.New("bearer token file is empty")
	}

	req.Header.Set("Authorization", "Bearer "+token)
	return nil
}

func expandPath(path string) string {
	return expandPathForOS(path, runtime.GOOS)
}

func expandPathForOS(path, goos string) string {
	if goos == "windows" {
		path = windowsEnvRE.ReplaceAllStringFunc(path, func(match string) string {
			return os.Getenv(match[1 : len(match)-1])
		})
	} else {
		path = os.ExpandEnv(path)
	}

	switch {
	case path == "~":
		if home, err := os.UserHomeDir(); err == nil {
			return home
		}
	case strings.HasPrefix(path, "~/"), strings.HasPrefix(path, `~\`):
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, path[2:])
		}
	}
	return path
}
