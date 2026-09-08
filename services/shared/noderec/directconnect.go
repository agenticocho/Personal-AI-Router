// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package noderec

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
)

// MethodSetDirectConnect is broker -> nvpair-cluster-manager: the local,
// sanitized direct-connect descriptor this node should publish on its
// authenticated roster response. It is a complete replacement snapshot, never a
// delta, and it is memory-only on the receiving side.
const MethodSetDirectConnect = "cluster:set-direct-connect"

const (
	maxDirectConnectBytes    = 4096
	maxDirectConnectServices = 32
)

// DirectConnect is the additive, authenticated direct-connect descriptor
// carried on the pinned-mTLS roster response. It names only an identity and a
// bounded set of known public service ports: no host, address, hostname, URL,
// path, token, header, PIN, certificate, key, or free-form metadata may appear,
// because the consumer's route host is its own configured seed and must never be
// redirected by a peer-supplied value.
type DirectConnect struct {
	HostUUID  string     `json:"hostUuid"`
	Principal string     `json:"principal"`
	Services  ServiceMap `json:"services"`
}

// DirectConnectParams is the internal broker -> cluster-manager envelope. The
// broker never supplies principal; the cluster-manager derives it from its own
// authenticated identity.
type DirectConnectParams struct {
	HostUUID string     `json:"hostUuid"`
	Services ServiceMap `json:"services"`
}

// UnmarshalJSON decodes a descriptor fail-closed. Oversized payloads, unknown
// or duplicate keys, non-object shapes, malformed values, and trailing data all
// reject the whole descriptor rather than yielding a partial one, so
// attacker-influenced routing state is never projected as authoritative.
func (d *DirectConnect) UnmarshalJSON(data []byte) error {
	if len(data) > maxDirectConnectBytes {
		return fmt.Errorf("direct-connect descriptor exceeds %d bytes", maxDirectConnectBytes)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	tok, err := dec.Token()
	if err != nil {
		return err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return fmt.Errorf("direct-connect descriptor must be an object")
	}
	var out DirectConnect
	seen := make(map[string]bool, 3)
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return err
		}
		key, ok := keyTok.(string)
		if !ok {
			return fmt.Errorf("direct-connect key must be a string")
		}
		if seen[key] {
			return fmt.Errorf("duplicate direct-connect key %q", key)
		}
		seen[key] = true
		switch key {
		case "hostUuid":
			if err := dec.Decode(&out.HostUUID); err != nil {
				return fmt.Errorf("direct-connect hostUuid: %w", err)
			}
		case "principal":
			if err := dec.Decode(&out.Principal); err != nil {
				return fmt.Errorf("direct-connect principal: %w", err)
			}
		case "services":
			services, err := decodeStrictServiceMap(dec)
			if err != nil {
				return err
			}
			out.Services = services
		default:
			return fmt.Errorf("unknown direct-connect field %q", key)
		}
	}
	if _, err := dec.Token(); err != nil {
		return err
	}
	var trailing any
	if err := dec.Decode(&trailing); err != io.EOF {
		if err == nil {
			return fmt.Errorf("trailing direct-connect data")
		}
		return err
	}
	*d = out
	return nil
}

// decodeStrictServiceMap decodes the descriptor's service map. Unlike the
// forward-compatible ServiceMap decoder it REJECTS an unknown service key: a
// descriptor is an authorization input, so an unrecognized entry means the
// sender and this consumer disagree about what is being authorized.
func decodeStrictServiceMap(dec *json.Decoder) (ServiceMap, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return nil, fmt.Errorf("direct-connect services must be an object")
	}
	out := make(ServiceMap)
	seen := make(map[string]bool)
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return nil, err
		}
		key, ok := keyTok.(string)
		if !ok {
			return nil, fmt.Errorf("direct-connect service key must be a string")
		}
		if seen[key] {
			return nil, fmt.Errorf("duplicate direct-connect service %q", key)
		}
		seen[key] = true
		if len(out) >= maxDirectConnectServices {
			return nil, fmt.Errorf("direct-connect services exceed %d entries", maxDirectConnectServices)
		}
		svc := ServiceKey(key)
		if !KnownServiceKey(svc) {
			return nil, fmt.Errorf("unknown direct-connect service %q", key)
		}
		var port int
		if err := dec.Decode(&port); err != nil {
			return nil, fmt.Errorf("direct-connect service %q port: %w", key, err)
		}
		if port < 1 || port > 65535 {
			return nil, fmt.Errorf("direct-connect service %q port must be 1..65535", key)
		}
		out[svc] = port
	}
	if _, err := dec.Token(); err != nil {
		return nil, err
	}
	return out, nil
}

// Clone deep-copies the descriptor so no caller shares the publisher's map.
func (d DirectConnect) Clone() DirectConnect {
	out := d
	out.Services = d.Services.Clone()
	return out
}

// BoundTo reports whether the descriptor is self-consistent AND bound to the
// exact principal the consumer authenticated. hostUuid, principal, and the
// pinned NodeUUID must all be the same value: a certificate belonging to
// another pinned member of the same cluster therefore cannot authenticate a
// descriptor for the selected peer.
func (d DirectConnect) BoundTo(principal string) bool {
	if principal == "" || d.HostUUID == "" || d.Principal == "" {
		return false
	}
	if d.HostUUID != principal || d.Principal != principal {
		return false
	}
	if len(d.Services) == 0 || len(d.Services) > maxDirectConnectServices {
		return false
	}
	for svc, port := range d.Services {
		if !KnownServiceKey(svc) || port < 1 || port > 65535 {
			return false
		}
	}
	return true
}

// SanitizeServices keeps only known public service keys with valid ports. It is
// the single filter the broker applies to its registration cache before the
// descriptor leaves the process.
func SanitizeServices(in ServiceMap) ServiceMap {
	out := make(ServiceMap, len(in))
	for svc, port := range in {
		if KnownServiceKey(svc) && port >= 1 && port <= 65535 {
			out[svc] = port
		}
	}
	return out
}
