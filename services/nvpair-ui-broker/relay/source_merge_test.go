// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package relay

import (
	"strconv"
	"testing"

	"nvpair-shared/noderec"
)

func scannerClaim(port int) noderec.DirectoryNode {
	return noderec.DirectoryNode{HostUUID: "h", Name: "scanner", IP: "192.168.1.2", ClusterUUID: "h", Trusted: true, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceNodeInfo: {Port: 14318}, noderec.ServiceLlamaCpp: {Port: port}}}
}
func manualClaim(port int) noderec.DirectoryNode {
	return noderec.DirectoryNode{HostUUID: "h", Name: "manual", IP: "10.0.0.2", ClusterUUID: "h", Trusted: true, Services: map[noderec.ServiceKey]noderec.ServiceStatus{noderec.ServiceLlamaCpp: {Port: port}}}
}
func TestScannerRemovalPreservesManualClaim(t *testing.T) {
	d := NewDirectory()
	d.ApplyManual(noderec.NotifyNodeUpdated, manualClaim(18435))
	d.Apply(noderec.NotifyNodeUpdated, scannerClaim(8081))
	d.Apply(noderec.NotifyNodeRemoved, noderec.DirectoryNode{HostUUID: "h"})
	n := d.Snapshot("")
	if len(n) != 1 || n[0].Name != "manual" || n[0].Services[noderec.ServiceLlamaCpp].Port != 18435 {
		t.Fatalf("manual claim lost: %#v", n)
	}
}
func TestManualRemovalPreservesScannerClaim(t *testing.T) {
	d := NewDirectory()
	d.Apply(noderec.NotifyNodeUpdated, scannerClaim(8081))
	d.ApplyManual(noderec.NotifyNodeUpdated, manualClaim(18435))
	d.ApplyManual(noderec.NotifyNodeRemoved, noderec.DirectoryNode{HostUUID: "h"})
	n := d.Snapshot("")
	if len(n) != 1 || n[0].Name != "scanner" || n[0].Services[noderec.ServiceLlamaCpp].Port != 8081 {
		t.Fatalf("scanner claim lost: %#v", n)
	}
}
func TestScannerWinsServiceConflicts(t *testing.T) {
	d := NewDirectory()
	d.ApplyManual(noderec.NotifyNodeUpdated, manualClaim(18435))
	d.Apply(noderec.NotifyNodeUpdated, scannerClaim(8081))
	n := d.Snapshot("")[0]
	if n.Services[noderec.ServiceLlamaCpp].Port != 8081 || n.Name != "scanner" {
		t.Fatalf("scanner did not win: %#v", n)
	}
}
func TestIdenticalProjectionDoesNotFanout(t *testing.T) {
	d := NewDirectory()
	calls := 0
	sub := &Subscriber{Send: func([]noderec.DirectoryNode) { calls++ }}
	d.Subscribe(sub)
	n := scannerClaim(8081)
	d.Apply(noderec.NotifyNodeUpdated, n)
	d.Apply(noderec.NotifyNodeUpdated, n)
	if calls != 1 {
		t.Fatalf("fanout=%d want 1", calls)
	}
}
func TestDynamicLlamaPortsRemainExact(t *testing.T) {
	for _, port := range []int{8081, 18435} {
		t.Run(strconv.Itoa(port), func(t *testing.T) {
			d := NewDirectory()
			d.ApplyManual(noderec.NotifyNodeUpdated, manualClaim(port))
			n := d.Snapshot(noderec.ServiceLlamaCpp)
			if len(n) != 1 || n[0].Services[noderec.ServiceLlamaCpp].Port != port {
				t.Fatalf("port changed: %#v", n)
			}
		})
	}
}
