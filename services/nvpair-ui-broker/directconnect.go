// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"nvpair-shared/noderec"
)

const directConnectPushTimeout = 5 * time.Second

// Snapshot delivery is serialized and coalesced. directConnectSeq is stamped at
// capture time and directConnectMu serializes the sends, so a slow older
// delivery can never land after — and therefore overwrite — a newer
// registration or withdrawal state.
var (
	directConnectMu  sync.Mutex
	directConnectSeq atomic.Uint64
)

// directConnectSnapshot builds the complete replacement snapshot. Pure function
// of identity plus the registration cache: known public service keys with valid
// ports only, no TXT metadata, credential, or filesystem path. The broker never
// supplies principal; the cluster-manager derives it from its own identity.
func directConnectSnapshot(nodeID string, regs []noderec.RegisterParams) noderec.DirectConnectParams {
	services := make(noderec.ServiceMap, len(regs))
	for _, p := range regs {
		if noderec.KnownServiceKey(p.Service) && p.Port >= 1 && p.Port <= 65535 {
			services[p.Service] = p.Port
		}
	}
	return noderec.DirectConnectParams{HostUUID: nodeID, Services: services}
}

// directConnectShouldSend reports whether a captured sequence is still the
// newest. An older snapshot is dropped rather than sent, because the newer one
// that superseded it is already queued behind the same lock.
func directConnectShouldSend(seq uint64) bool {
	return seq == directConnectSeq.Load()
}

func (b *Broker) pushDirectConnectToClusterManager() {
	b.pushDirectConnectVia(b.getClusterMgr())
}

// pushDirectConnectVia sends through an explicit handle so a spawn can replay
// through the process it just created, without depending on the supervised
// handle already being published by getClusterMgr.
func (b *Broker) pushDirectConnectVia(cm *clusterManagerProcess) {
	if cm == nil {
		return
	}
	seq := directConnectSeq.Add(1)
	params, err := json.Marshal(directConnectSnapshot(b.nodeID, b.regCache.Snapshot()))
	if err != nil {
		slog.Warn("marshal direct-connect snapshot failed", "err", err)
		return
	}
	go func() {
		directConnectMu.Lock()
		defer directConnectMu.Unlock()
		if !directConnectShouldSend(seq) {
			return
		}
		ctx, cancel := context.WithTimeout(context.Background(), directConnectPushTimeout)
		defer cancel()
		if _, rpcErr, err := cm.Call(ctx, noderec.MethodSetDirectConnect, params); err != nil || rpcErr != nil {
			slog.Warn("push direct-connect snapshot failed", "err", err, "rpcErr", rpcErr)
		}
	}()
}
