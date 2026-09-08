// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"nvpair-shared/applog"
	"nvpair-shared/noderec"
	"testing"
)

func TestServiceSnapshotControlAndResponse(t *testing.T) {
	s := &serviceSnapshot{}
	raw, _ := json.Marshal(noderec.ServiceMapParams{Services: noderec.ServiceMap{noderec.ServiceLlamaCpp: 8081}})
	handleServiceSnapshot(applog.StdinMessage{Method: noderec.MethodSetServices, Params: raw}, s)
	body := attachServices([]byte(`{"GPUs":[],"telemetryValid":false,"msSince":0}`), s.get())
	var got NodeInfoResponse
	if err := json.Unmarshal(body, &got); err != nil || got.Services[noderec.ServiceLlamaCpp] != 8081 {
		t.Fatalf("%v %#v", err, got.Services)
	}
}
