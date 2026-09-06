// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"

	"nvpair-shared/appdir"
	"nvpair-shared/httpauth"
)

const maxLocalEngineOverrideBytes = 64 * 1024

var errLocalBackendAuthentication = errors.New(
	"local inference backend is unavailable",
)

// localAuthRoundTripper runs after ReverseProxy has removed hop-by-hop
// headers. It clones the outbound request, resolves the current destination
// credential, and replaces Authorization only for this local backend attempt.
type localAuthRoundTripper struct {
	base http.RoundTripper
}

func (transport localAuthRoundTripper) RoundTrip(
	req *http.Request,
) (*http.Response, error) {
	outbound := req.Clone(req.Context())
	outbound.Header = req.Header.Clone()

	if err := applyLocalBackendAuthorization(outbound); err != nil {
		return nil, errLocalBackendAuthentication
	}

	base := transport.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(outbound)
}

func withLocalBackendAuthorization(
	base http.RoundTripper,
) http.RoundTripper {
	return localAuthRoundTripper{base: base}
}

func isLocalBackendAuthenticationError(err error) bool {
	return errors.Is(err, errLocalBackendAuthentication)
}

// localBearerTokenFile reads only the local llama.cpp runtime credential
// reference. Missing configuration means the backend needs no authorization.
func localBearerTokenFile() (string, error) {
	path, err := appdir.Path("engines", "llamacpp.json")
	if err != nil {
		return "", errors.New("local backend configuration is unavailable")
	}

	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", errors.New("local backend configuration is unavailable")
	}
	defer file.Close()

	info, err := file.Stat()
	if err != nil || !info.Mode().IsRegular() {
		return "", errors.New("local backend configuration is invalid")
	}

	data, err := io.ReadAll(
		io.LimitReader(file, maxLocalEngineOverrideBytes+1),
	)
	if err != nil {
		return "", errors.New("local backend configuration is unreadable")
	}
	if len(data) > maxLocalEngineOverrideBytes {
		return "", errors.New("local backend configuration is too large")
	}

	var config struct {
		Runtime struct {
			BearerTokenFile string `json:"bearer_token_file"`
		} `json:"runtime"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return "", errors.New("local backend configuration is invalid")
	}
	return config.Runtime.BearerTokenFile, nil
}

// applyLocalBackendAuthorization resolves both configuration and token for
// every request. Errors are deliberately generic so paths never reach logs,
// responses, workload events, or peer-facing contracts.
func applyLocalBackendAuthorization(req *http.Request) error {
	tokenFile, err := localBearerTokenFile()
	if err != nil {
		return errors.New("local backend authentication is unavailable")
	}
	if err := httpauth.ApplyBearerTokenFile(req, tokenFile); err != nil {
		return errors.New("local backend authentication is unavailable")
	}
	return nil
}
