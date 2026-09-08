// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestDirectConnectRejectsMalformedDescriptors(t *testing.T) {
	cases := map[string]string{
		"host field":        `{"hostUuid":"a","principal":"a","host":"10.0.0.9","services":{"ll":8081}}`,
		"address field":     `{"hostUuid":"a","principal":"a","address":"10.0.0.9","services":{"ll":8081}}`,
		"url field":         `{"hostUuid":"a","principal":"a","url":"https://x/","services":{"ll":8081}}`,
		"token field":       `{"hostUuid":"a","principal":"a","token":"t","services":{"ll":8081}}`,
		"duplicate service": `{"hostUuid":"a","principal":"a","services":{"ll":8081,"ll":18435}}`,
		"duplicate key":     `{"hostUuid":"a","hostUuid":"b","principal":"a","services":{"ll":8081}}`,
		"unknown service":   `{"hostUuid":"a","principal":"a","services":{"zz":8081}}`,
		"zero port":         `{"hostUuid":"a","principal":"a","services":{"ll":0}}`,
		"high port":         `{"hostUuid":"a","principal":"a","services":{"ll":65536}}`,
		"string port":       `{"hostUuid":"a","principal":"a","services":{"ll":"8081"}}`,
		"not an object":     `["a"]`,
		"trailing data":     `{"hostUuid":"a","principal":"a","services":{"ll":8081}} {}`,
		"oversized":         `{"hostUuid":"` + strings.Repeat("a", 5000) + `","principal":"a","services":{"ll":8081}}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			var d DirectConnect
			if err := json.Unmarshal([]byte(body), &d); err == nil {
				t.Fatalf("accepted %s: %#v", name, d)
			}
		})
	}
}

func TestDirectConnectBindsExactPrincipal(t *testing.T) {
	var d DirectConnect
	if err := json.Unmarshal([]byte(`{"hostUuid":"host-a","principal":"host-a","services":{"ll":8081,"em":14319,"ec":14322}}`), &d); err != nil {
		t.Fatal(err)
	}
	if !d.BoundTo("host-a") {
		t.Fatal("valid descriptor rejected")
	}
	for _, other := range []string{"", "host-b", "cluster-a"} {
		if d.BoundTo(other) {
			t.Fatalf("descriptor bound to %q", other)
		}
	}
	if d.Services[ServiceLlamaCpp] != 8081 {
		t.Fatalf("dynamic llama.cpp port normalized: %d", d.Services[ServiceLlamaCpp])
	}
	mismatched := DirectConnect{HostUUID: "host-a", Principal: "host-b", Services: ServiceMap{ServiceLlamaCpp: 18435}}
	if mismatched.BoundTo("host-a") || mismatched.BoundTo("host-b") {
		t.Fatal("hostUuid/principal mismatch accepted")
	}
	empty := DirectConnect{HostUUID: "host-a", Principal: "host-a"}
	if empty.BoundTo("host-a") {
		t.Fatal("empty service map accepted")
	}
}

func TestSanitizeServicesDropsUnknownAndInvalid(t *testing.T) {
	got := SanitizeServices(ServiceMap{ServiceLlamaCpp: 18435, ServiceKey("zz"): 9000, ServiceOllama: 0, ServiceEngineControl: 14322})
	if len(got) != 2 || got[ServiceLlamaCpp] != 18435 || got[ServiceEngineControl] != 14322 {
		t.Fatalf("sanitize=%#v", got)
	}
}
