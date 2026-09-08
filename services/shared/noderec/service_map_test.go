// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"encoding/json"
	"testing"
)

func TestServiceMapStrictValidation(t *testing.T) {
	var m ServiceMap
	if err := json.Unmarshal([]byte(`{"ll":8081,"future":{"shape":"ignored"}}`), &m); err != nil || m[ServiceLlamaCpp] != 8081 || len(m) != 1 {
		t.Fatalf("valid map: %v %#v", err, m)
	}
	for _, raw := range []string{`{"ll":0}`, `{"ll":65536}`, `{"ll":8081,"ll":8082}`, `[]`, `{"ll":8081} true`} {
		if json.Unmarshal([]byte(raw), &m) == nil {
			t.Errorf("accepted %s", raw)
		}
	}
}
