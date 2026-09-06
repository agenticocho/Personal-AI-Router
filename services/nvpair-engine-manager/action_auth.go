// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"net/http"

	"nvpair-shared/httpauth"
)

const maxBearerTokenFileBytes = httpauth.MaxBearerTokenFileBytes

func applyActionHTTPAuthorization(
	req *http.Request,
	tokenFile string,
) error {
	return httpauth.ApplyBearerTokenFile(req, tokenFile)
}
