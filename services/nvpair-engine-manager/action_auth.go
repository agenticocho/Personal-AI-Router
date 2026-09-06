// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"runtime"
	"strings"
)

const maxBearerTokenFileBytes = 16 * 1024

func applyActionHTTPAuthorization(req *http.Request, tokenFile string) error {
	if strings.TrimSpace(tokenFile) == "" {
		return nil
	}

	tokenPath := expandPath(tokenFile)
	file, err := os.Open(tokenPath)
	if err != nil {
		return fmt.Errorf("open bearer token file %q: %w", tokenPath, err)
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("stat bearer token file %q: %w", tokenPath, err)
	}
	if !info.Mode().IsRegular() {
		return fmt.Errorf("bearer token file %q is not a regular file", tokenPath)
	}
	if runtime.GOOS != "windows" && info.Mode().Perm()&0o077 != 0 {
		return fmt.Errorf(
			"bearer token file %q must not be accessible by group or other",
			tokenPath,
		)
	}

	data, err := io.ReadAll(
		io.LimitReader(file, maxBearerTokenFileBytes+1),
	)
	if err != nil {
		return fmt.Errorf("read bearer token file %q: %w", tokenPath, err)
	}
	if len(data) > maxBearerTokenFileBytes {
		return fmt.Errorf(
			"bearer token file %q exceeds %d bytes",
			tokenPath,
			maxBearerTokenFileBytes,
		)
	}

	token := strings.TrimSpace(string(data))
	if token == "" {
		return fmt.Errorf("bearer token file %q is empty", tokenPath)
	}
	if strings.ContainsAny(token, "\r\n") {
		return fmt.Errorf("bearer token file %q contains multiple lines", tokenPath)
	}

	req.Header.Set("Authorization", "Bearer "+token)
	return nil
}
