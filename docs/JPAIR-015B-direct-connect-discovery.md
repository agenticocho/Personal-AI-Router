<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Direct-connect discovery fallback

JPAIR-015B lets already-paired nodes keep cluster operations when multicast DNS
does not cross a direct Ethernet link. It extends the existing manual-node seed
path and the existing pinned-mTLS roster endpoint. It adds no public endpoint,
no pairing protocol change, no relay transport, and no background scan.

## Data path

1. The broker sends `nvpair-cluster-manager` a complete replacement
   direct-connect snapshot after cluster-manager spawn or restart and after every
   real service registration or withdrawal. The snapshot carries `hostUuid` and a
   sanitized service map only. Deliveries are sequenced and serialized, so an
   older snapshot can never overwrite newer registration state.
2. The cluster-manager derives `principal` from its own authenticated identity,
   requires `hostUuid == principal == identity.NodeUUID`, and holds the
   descriptor in memory only. It is never persisted, never loaded from disk, and
   never enters `members.json` or the pin store.
3. The descriptor is returned only from the existing trusted mTLS handler for
   `GET /v1/cluster/roster`, as an additive optional field.
4. A manual-node probe reads plaintext `/v1/node-info` for exactly two hints:
   the principal to select a pin for, and optionally the cluster-manager port.
5. The prober connects to its own configured seed address using the pin selected
   for that principal, confirms the presented leaf names that principal, and
   fetches the roster over that same authenticated connection.

## Classification

The probe result is tri-state, and the three states are not interchangeable.

| Case | Condition | Result |
| --- | --- | --- |
| 1 | No `clusterUuid`, and this entry has never reported one | `descriptor_absent`; legacy standalone behavior preserved, no valid service map, no authenticated relay claim |
| 2 | Nonempty `clusterUuid`, then missing local cluster state, missing pin, TLS failure, certificate mismatch, roster failure, malformed descriptor, or identity mismatch | `authentication_failed`; all legacy and authenticated claims suppressed |
| 3 | Pinned-mTLS roster succeeds but omits `directConnect` | `descriptor_absent`; the specified old-peer partial behavior |
| 4 | This entry previously reported a nonempty `clusterUuid` and a later plaintext response omits or empties it | `authentication_failed`; treated as a downgrade attempt, never reverted to standalone behavior |

Downgrade memory is process-local. It records only the last nonempty cluster
identity an entry asserted, is never written to disk, and is cleared by exactly
one event: explicit deletion of the manual entry. A probe failure, an
authentication failure, a stripped annotation, or a vanished status or relay
claim all retain it; a worker restart clears it with the process, after which
the next probe re-earns it.

## Security

Plaintext node-info never authorizes relay claims, engine inventory, engine
control, llama.cpp routing, or trusted-node status. A certificate for another
pinned member of the same cluster does not authenticate the selected peer.

The descriptor wire shape admits only `hostUuid`, `principal`, and `services`.
No host, address, hostname, URL, path, token, cookie, header, PIN, certificate,
key, or arbitrary metadata field is representable, so a peer cannot redirect the
consumer. The configured manual seed remains the sole route host, and there is no
plaintext fallback after any authenticated failure.

## Routing

The manual worker makes no direct plaintext request to a reported llama.cpp
port. It projects authenticated service locations into `relay.Directory`, and
existing subscribers consume them unchanged: `em` for model inventory, `ec` for
authenticated engine status and control, and `ll` for llama.cpp routing. The
llama.cpp proxy's mTLS and destination-local bearer behavior is unchanged.

## Merge and compatibility

Scanner claims own identity and conflicting service facts. An authenticated
manual claim contributes only missing services and puts the configured seed
first in the candidate address list. Removing either source removes only that
claim; the final source's removal deletes the node. Identical projections do not
fan out.

Peers without `directConnect` keep their existing partial behavior, and old
consumers ignore the additive field. Dynamic ports are never normalized: `ll=8081`
and `ll=18435` are relayed exactly. mDNS remains primary when available, and
existing paired clusters need no migration or re-pair.
