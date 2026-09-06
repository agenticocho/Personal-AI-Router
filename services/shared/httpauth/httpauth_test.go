// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package httpauth

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func privateTokenFile(t *testing.T, path, token string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(token), 0o600); err != nil {
		t.Fatal(err)
	}
}

func newRequest() *http.Request {
	return httptest.NewRequest(http.MethodGet, "http://127.0.0.1/", nil)
}

func TestApplyBearerTokenFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "token")
	privateTokenFile(t, path, "expected-token\n")

	request := newRequest()
	if err := ApplyBearerTokenFile(request, path); err != nil {
		t.Fatal(err)
	}
	if request.Header.Get("Authorization") != "Bearer expected-token" {
		t.Fatal("request did not receive expected bearer authorization")
	}
}

func TestApplyBearerTokenFileOmitted(t *testing.T) {
	request := newRequest()
	request.Header.Set("Authorization", "Bearer caller-token")

	if err := ApplyBearerTokenFile(request, ""); err != nil {
		t.Fatal(err)
	}
	if request.Header.Get("Authorization") != "Bearer caller-token" {
		t.Fatal("empty path unexpectedly modified Authorization")
	}
}

func TestApplyBearerTokenFileExpandsEnvironment(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "token")
	privateTokenFile(t, path, "expanded-token")

	t.Setenv("HTTPAUTH_TEST_DIR", dir)
	var configured string
	if runtime.GOOS == "windows" {
		configured = `%HTTPAUTH_TEST_DIR%\token`
	} else {
		configured = "$HTTPAUTH_TEST_DIR/token"
	}

	request := newRequest()
	if err := ApplyBearerTokenFile(request, configured); err != nil {
		t.Fatal(err)
	}
	if request.Header.Get("Authorization") != "Bearer expanded-token" {
		t.Fatal("expanded token path did not authenticate request")
	}
}

func TestExpandPathForOSForms(t *testing.T) {
	t.Setenv("HTTPAUTH_TEST_VALUE", "expanded")

	if got := expandPathForOS(
		`%HTTPAUTH_TEST_VALUE%\token`,
		"windows",
	); got != `expanded\token` {
		t.Fatalf("Windows environment expansion = %q", got)
	}
	if got := expandPathForOS(
		"$HTTPAUTH_TEST_VALUE/token",
		"linux",
	); got != "expanded/token" {
		t.Fatalf("Unix environment expansion = %q", got)
	}
}

func TestApplyBearerTokenFileRotation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "token")
	privateTokenFile(t, path, "first-token")

	first := newRequest()
	if err := ApplyBearerTokenFile(first, path); err != nil {
		t.Fatal(err)
	}
	if first.Header.Get("Authorization") != "Bearer first-token" {
		t.Fatal("first request did not use initial token")
	}

	privateTokenFile(t, path, "second-token")
	second := newRequest()
	if err := ApplyBearerTokenFile(second, path); err != nil {
		t.Fatal(err)
	}
	if second.Header.Get("Authorization") != "Bearer second-token" {
		t.Fatal("second request did not use rotated token")
	}
}

func TestApplyBearerTokenFileRejectsPermissions(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("Unix permission test")
	}

	path := filepath.Join(t.TempDir(), "token")
	privateTokenFile(t, path, "secret-value")
	if err := os.Chmod(path, 0o644); err != nil {
		t.Fatal(err)
	}

	request := newRequest()
	err := ApplyBearerTokenFile(request, path)
	if err == nil || !strings.Contains(err.Error(), "group or other") {
		t.Fatalf("expected insecure-permission rejection, got %v", err)
	}
	if request.Header.Get("Authorization") != "" {
		t.Fatal("rejected credential modified Authorization")
	}
}

func TestApplyBearerTokenFileRejectsInvalidContent(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    string
	}{
		{name: "empty", content: "", want: "empty"},
		{name: "newline-only", content: "\n", want: "empty"},
		{name: "multiline", content: "one\ntwo\n", want: "multiple lines"},
		{name: "extra trailing newline", content: "one\n\n", want: "multiple lines"},
		{name: "NUL", content: "one\x00two", want: "NUL"},
		{
			name:    "oversized",
			content: strings.Repeat("x", MaxBearerTokenFileBytes+1),
			want:    "exceeds",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "token")
			privateTokenFile(t, path, test.content)

			request := newRequest()
			err := ApplyBearerTokenFile(request, path)
			if err == nil || !strings.Contains(err.Error(), test.want) {
				t.Fatalf("expected %q rejection, got %v", test.want, err)
			}
			if request.Header.Get("Authorization") != "" {
				t.Fatal("rejected credential modified Authorization")
			}
		})
	}
}

func TestApplyBearerTokenFileRejectsNonRegular(t *testing.T) {
	request := newRequest()
	err := ApplyBearerTokenFile(request, t.TempDir())
	if err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Fatalf("expected non-regular-file rejection, got %v", err)
	}
}

func TestApplyBearerTokenFileErrorsDoNotDisclose(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "credential-secret-name")
	privateTokenFile(t, path, "do-not-disclose")
	if runtime.GOOS != "windows" {
		if err := os.Chmod(path, 0o644); err != nil {
			t.Fatal(err)
		}
	} else {
		if err := os.Remove(path); err != nil {
			t.Fatal(err)
		}
	}

	err := ApplyBearerTokenFile(newRequest(), path)
	if err == nil {
		t.Fatal("expected credential-file error")
	}
	for _, forbidden := range []string{
		path,
		dir,
		"credential-secret-name",
		"do-not-disclose",
	} {
		if strings.Contains(err.Error(), forbidden) {
			t.Fatal("credential error disclosed sensitive material")
		}
	}
}
