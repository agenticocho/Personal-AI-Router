// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
    getModularSupervisor,
    parseListModelNames
} from '../../src/electron/service-bridge/modular-supervisor'
import { getModularBridgeState } from '../../src/electron/service-bridge/modular-state'

vi.mock('../../src/electron/service-bridge/broadcaster', () => ({
    emitBridgePush: vi.fn()
}))

const MODEL_A = 'DeepSeek-V4-Pro-Qwen3.5-4B-MTP-Q4_K_M'
const MODEL_B = 'DeepSeek-V4-Pro-Qwen3.5-4B-MTP-Q4KM'
const SELF_ID = 'jpair-014z-test-self'

type Method = (...args: unknown[]) => unknown

function methodOf(target: object, name: string): Method {
    const value = Reflect.get(target, name)
    if (typeof value !== 'function') {
        throw new TypeError(`${name} is not callable`)
    }
    return (...args: unknown[]) => Reflect.apply(value, target, args)
}

function mapOf(target: object, name: string): Map<unknown, unknown> {
    const value = Reflect.get(target, name)
    if (!(value instanceof Map)) {
        throw new TypeError(`${name} is not a Map`)
    }
    return value
}

function bridgeOf(supervisor: object): object {
    const bridges = mapOf(supervisor, 'localBridges')
    const bridge = bridges.get('llamacpp')
    if (typeof bridge !== 'object' || bridge === null) {
        throw new TypeError('llamacpp bridge missing')
    }
    return bridge
}

function payloadModels(call: unknown[]): string[] {
    const value = call[2]
    if (typeof value !== 'object' || value === null) {
        throw new TypeError('node/add-manual payload missing')
    }

    const models = Reflect.get(value, 'models')
    if (!Array.isArray(models) || !models.every(model => typeof model === 'string')) {
        throw new TypeError('node/add-manual models missing or invalid')
    }

    return models
}

function payloadMatches(
    call: unknown[],
    expectedId: string,
    expectedPort: number,
    expectedModels: string[]
): boolean {
    const value = call[2]
    if (typeof value !== 'object' || value === null) {
        return false
    }

    const id = Reflect.get(value, 'id')
    const port = Reflect.get(value, 'port')
    const models = Reflect.get(value, 'models')

    return (
        id === expectedId &&
        port === expectedPort &&
        Array.isArray(models) &&
        models.every(model => typeof model === 'string') &&
        models.length === expectedModels.length &&
        models.every((model, index) => model === expectedModels[index])
    )
}

describe('parseListModelNames strict inventory shapes', () => {
    const exact = 'DeepSeek-V4-Pro-Qwen3.5-4B-MTP-Q4_K_M'
    const compact = 'DeepSeek-V4-Pro-Qwen3.5-4B-MTP-Q4KM'

    it('accepts exact OpenAI data identifiers', () => {
        expect(parseListModelNames({ data: [{ id: exact }] })).toEqual([exact])
    })

    it('accepts authoritative empty arrays', () => {
        expect(parseListModelNames({ data: [] })).toEqual([])
        expect(parseListModelNames({ models: [] })).toEqual([])
    })

    it('preserves models name and key compatibility', () => {
        expect(
            parseListModelNames({
                models: [{ name: exact }, { key: compact }]
            })
        ).toEqual([exact, compact])
    })

    it('skips malformed rows when a usable identifier remains', () => {
        expect(
            parseListModelNames({
                data: [null, { id: 42 }, { id: exact }]
            })
        ).toEqual([exact])
    })

    it('rejects nonempty arrays with no usable identifiers', () => {
        expect(() => parseListModelNames({ data: [null, { id: 42 }] })).toThrow(
            'contains no usable identifiers'
        )
    })

    it('rejects missing, null, and wrong-typed arrays', () => {
        expect(() => parseListModelNames({})).toThrow('missing both models and data arrays')
        expect(() => parseListModelNames({ data: null })).toThrow('data field must be an array')
        expect(() => parseListModelNames({ models: 'invalid' })).toThrow(
            'models field must be an array'
        )
    })

    it('rejects conflicting dual shapes', () => {
        expect(() =>
            parseListModelNames({
                models: [{ name: exact }],
                data: [{ id: compact }]
            })
        ).toThrow('conflicting models and data arrays')
    })

    it('accepts identical deterministic dual shapes', () => {
        expect(
            parseListModelNames({
                models: [{ name: exact }],
                data: [{ id: exact }]
            })
        ).toEqual([exact])
    })

    it('preserves punctuation-distinct identifiers byte-for-byte', () => {
        expect(
            parseListModelNames({
                data: [{ id: compact }, { id: exact }]
            })
        ).toEqual([compact, exact])
    })
})

describe('local model capability propagation', () => {
    const supervisor = getModularSupervisor()
    const state = getModularBridgeState()

    let originalCallProxy: unknown
    let originalCallProcess: unknown
    let proxyCalls: unknown[][]
    let callProxy: ReturnType<typeof vi.fn>

    const invoke = (name: string, ...args: unknown[]): unknown =>
        methodOf(supervisor, name)(...args)

    const settle = async (): Promise<void> => {
        const chain = Reflect.get(bridgeOf(supervisor), 'reconcileChain')
        if (!(chain instanceof Promise)) {
            throw new TypeError('reconcile chain missing')
        }
        await chain
    }

    const commitInventory = async (models: string[]): Promise<void> => {
        const generation = invoke('beginModelRefresh', 'llamacpp')
        if (typeof generation !== 'number') {
            throw new TypeError('invalid inventory generation')
        }
        invoke('commitModelInventory', 'llamacpp', models, generation)
        await settle()
    }

    const nodeAdds = (): unknown[][] =>
        proxyCalls.filter(call => call[0] === 'llamacpp' && call[1] === 'node/add-manual')

    beforeEach(async () => {
        originalCallProxy = Reflect.get(supervisor, 'callProxy')
        originalCallProcess = Reflect.get(supervisor, 'callProcess')

        proxyCalls = []
        callProxy = vi.fn(async (...args: unknown[]) => {
            proxyCalls.push(args)
            return { added: true }
        })

        Reflect.set(supervisor, 'callProxy', callProxy)
        Reflect.set(
            supervisor,
            'callProcess',
            vi.fn(async () => ({}))
        )

        mapOf(supervisor, 'processes').clear()
        mapOf(supervisor, 'processes').set('broker', {})
        mapOf(supervisor, 'processes').set('engine-manager', {})
        mapOf(supervisor, 'localBridges').clear()
        mapOf(supervisor, 'modelRefreshGenerations').clear()
        mapOf(state, 'localManagerModels').clear()

        Reflect.set(state, 'selfId', SELF_ID)
        const ports = Reflect.get(state, 'proxyPorts')
        if (typeof ports !== 'object' || ports === null) {
            throw new TypeError('proxy ports missing')
        }
        Reflect.set(ports, 'llamacpp', 18435)

        invoke('updateLocalNodeBridgeFromEngineState', {
            engine: 'llamacpp',
            installed: true,
            running: true,
            healthy: true,
            port: 18434
        })
        await settle()
        proxyCalls = []
    })

    afterEach(async () => {
        await settle()
        Reflect.set(supervisor, 'callProxy', originalCallProxy)
        Reflect.set(supervisor, 'callProcess', originalCallProcess)
        mapOf(supervisor, 'processes').clear()
        mapOf(supervisor, 'localBridges').clear()
        mapOf(supervisor, 'modelRefreshGenerations').clear()
        mapOf(state, 'localManagerModels').clear()
        Reflect.set(state, 'selfId', null)
        vi.restoreAllMocks()
    })

    it('bridges the exact authoritative Q4_K_M model identifier', async () => {
        await commitInventory([MODEL_A])

        await vi.waitFor(() => expect(nodeAdds()).toHaveLength(1))
        expect(payloadModels(nodeAdds()[0])).toEqual([MODEL_A])
    })

    it('keeps Q4KM and Q4_K_M as distinct identifiers', async () => {
        expect(
            parseListModelNames({
                models: [{ name: MODEL_B }, { name: MODEL_A }]
            })
        ).toEqual([MODEL_B, MODEL_A])

        await commitInventory([MODEL_B, MODEL_A])
        expect(payloadModels(nodeAdds()[0])).toEqual([MODEL_B, MODEL_A])
    })

    it('re-bridges a changed inventory with unchanged self ID and port', async () => {
        await commitInventory([MODEL_A])
        await commitInventory([MODEL_B])

        expect(nodeAdds()).toHaveLength(2)
        expect(payloadModels(nodeAdds()[0])).toEqual([MODEL_A])
        expect(payloadModels(nodeAdds()[1])).toEqual([MODEL_B])
    })

    it('does not issue a redundant update for identical inventory', async () => {
        await commitInventory([MODEL_A])
        proxyCalls = []

        await commitInventory([MODEL_A])
        await settle()

        expect(nodeAdds()).toHaveLength(0)
    })

    it('propagates an authoritative empty inventory', async () => {
        await commitInventory([MODEL_A])
        proxyCalls = []

        await commitInventory([])

        expect(nodeAdds()).toHaveLength(1)
        expect(payloadModels(nodeAdds()[0])).toEqual([])
        expect(Reflect.get(bridgeOf(supervisor), 'bridgedModelNames')).toEqual([])
    })

    it('cannot let stale in-flight completion replace newer inventory', async () => {
        let releaseFirst = (): void => undefined
        let markFirstStarted = (): void => undefined

        const firstStarted = new Promise<void>(resolve => {
            markFirstStarted = resolve
        })

        callProxy.mockImplementationOnce(
            (...args: unknown[]) =>
                new Promise(resolve => {
                    proxyCalls.push(args)
                    releaseFirst = () => resolve({ added: true })
                    markFirstStarted()
                })
        )

        const oldGeneration = invoke('beginModelRefresh', 'llamacpp')
        if (typeof oldGeneration !== 'number') {
            throw new TypeError('invalid old generation')
        }
        invoke('commitModelInventory', 'llamacpp', [MODEL_A], oldGeneration)

        await firstStarted
        expect(nodeAdds()).toHaveLength(1)

        const newGeneration = invoke('beginModelRefresh', 'llamacpp')
        if (typeof newGeneration !== 'number') {
            throw new TypeError('invalid new generation')
        }
        invoke('commitModelInventory', 'llamacpp', [MODEL_B], newGeneration)

        releaseFirst()
        await settle()

        expect(nodeAdds()).toHaveLength(2)
        expect(payloadModels(nodeAdds()[1])).toEqual([MODEL_B])
        expect(Reflect.get(bridgeOf(supervisor), 'bridgedModelNames')).toEqual([MODEL_B])
    })

    it('bridges hydrated inventory after self ID becomes known', async () => {
        Reflect.set(state, 'selfId', null)
        mapOf(supervisor, 'localBridges').clear()
        proxyCalls = []

        invoke('updateLocalNodeBridgeFromEngineState', {
            engine: 'llamacpp',
            installed: true,
            running: true,
            healthy: true,
            port: 18434
        })
        await commitInventory([MODEL_A])

        expect(nodeAdds()).toHaveLength(0)

        Reflect.set(state, 'selfId', SELF_ID)
        const reconciliation = invoke('reconcileLocalNodeBridge', 'llamacpp')
        if (!(reconciliation instanceof Promise)) {
            throw new TypeError('reconciliation did not return Promise')
        }
        await reconciliation
        await settle()

        expect(nodeAdds()).toHaveLength(1)
        expect(payloadModels(nodeAdds()[0])).toEqual([MODEL_A])
    })

    it('re-sends capabilities after proxy restart state is cleared', async () => {
        await commitInventory([MODEL_A])
        proxyCalls = []

        const bridge = bridgeOf(supervisor)
        Reflect.set(bridge, 'bridgedId', '')
        Reflect.set(bridge, 'bridgedPort', 0)
        Reflect.set(bridge, 'bridgedModelNames', [])

        const reconciliation = invoke('reconcileLocalNodeBridge', 'llamacpp')
        if (!(reconciliation instanceof Promise)) {
            throw new TypeError('reconciliation did not return Promise')
        }
        await reconciliation

        expect(nodeAdds()).toHaveLength(1)
        expect(payloadMatches(nodeAdds()[0], SELF_ID, 18434, [MODEL_A])).toBe(true)
    })

    it('commits raw OpenAI inventory into node/add-manual', async () => {
        const exact = 'DeepSeek-V4-Pro-Qwen3.5-4B-MTP-Q4_K_M'
        const rawActionResponse = {
            object: 'list',
            data: [{ id: exact, object: 'model', owned_by: 'llamacpp' }]
        }

        await commitInventory(parseListModelNames(rawActionResponse))

        expect(nodeAdds()).toHaveLength(1)
        expect(payloadModels(nodeAdds()[0])).toEqual([exact])
    })
})
