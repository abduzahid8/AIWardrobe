/**
 * wardrobeGraph.ts — Wardrobe graph (Phase 3 §33).
 *
 * Graph-like domain relationships WITHOUT a graph database: adjacency is
 * built on demand from wardrobe items + outfit memory + DNA, with hard caps
 * (no unbounded traversal — BFS depth <= 2, edge cap per node).
 */

import type { OutfitMemoryEntry, WardrobeGraphEdge } from '../../types/styleOS';

export interface GraphItemView {
  id: string;
  category: string;
  subCategory?: string;
  primaryColor?: string;
  brand?: string;
  occasions?: string[];
  seasons?: string[];
}

export interface WardrobeGraph {
  nodes: string[];
  edges: WardrobeGraphEdge[];
}

const MAX_EDGES_PER_NODE = 12;
const MAX_EDGES_TOTAL = 2000;

function roleKeyOf(item: GraphItemView): string {
  return `${item.category}:${(item.subCategory || 'other').toLowerCase()}:${(item.primaryColor || 'any').toLowerCase()}`;
}

function nodesOf(items: readonly GraphItemView[], memory: readonly OutfitMemoryEntry[]): string[] {
  const nodes = new Set<string>();
  for (const item of items) {
    nodes.add(`garment:${item.id}`);
    nodes.add(`role:${roleKeyOf(item)}`);
    if (item.primaryColor) nodes.add(`color:${item.primaryColor.toLowerCase()}`);
    if (item.brand) nodes.add(`brand:${item.brand.toLowerCase()}`);
    for (const o of item.occasions || []) nodes.add(`context:${o.toLowerCase()}`);
  }
  for (const m of memory) {
    nodes.add(`outfit:${m.outfitId}`);
    for (const id of m.itemIds) nodes.add(`garment:${id}`);
    if (m.contextId) nodes.add(`context:${m.contextId.toLowerCase()}`);
  }
  return [...nodes].slice(0, 4000);
}

/**
 * Build edges from co-wear evidence + role similarity. Deterministic;
 * capped so render paths never do O(N^2) work.
 */
export function buildWardrobeGraph(
  items: readonly GraphItemView[],
  memory: readonly OutfitMemoryEntry[],
): WardrobeGraph {
  const edges: WardrobeGraphEdge[] = [];
  const perNode = new Map<string, number>();
  const push = (e: WardrobeGraphEdge) => {
    if (edges.length >= MAX_EDGES_TOTAL) return;
    const c = perNode.get(e.from) || 0;
    if (c >= MAX_EDGES_PER_NODE) return;
    perNode.set(e.from, c + 1);
    edges.push(e);
  };
  const byId = new Map(items.map((i) => [i.id, i]));
  // worn_with from confirmed memory.
  for (const m of memory.slice(-500)) {
    if (!m.confirmed) continue;
    const ids = m.itemIds.slice(0, 6);
    for (let a = 0; a < ids.length; a++) {
      for (let b = a + 1; b < ids.length; b++) {
        push({
          from: `garment:${ids[a]}`,
          to: `garment:${ids[b]}`,
          kind: m.rating === 'loved' || m.rating === 'good' ? 'works_with' : 'worn_with',
          weight: m.rating === 'loved' ? 1 : m.rating === 'good' ? 0.8 : 0.5,
          evidence: `Worn together ${m.wornDate}${m.rating ? ` rated ${m.rating}` : ''}`,
        });
      }
      push({
        from: `garment:${ids[a]}`,
        to: `outfit:${m.outfitId}`,
        kind: 'worn_with',
        weight: 0.6,
        evidence: `Part of outfit worn ${m.wornDate}`,
      });
      if (m.contextId)
        push({
          from: `garment:${ids[a]}`,
          to: `context:${m.contextId.toLowerCase()}`,
          kind: 'preferred_for',
          weight: 0.5,
          evidence: `Worn for ${m.contextId}`,
        });
    }
  }
  // Role similarity edges (bounded: same category+color only).
  const seen = new Set<string>();
  for (const item of items.slice(0, 500)) {
    const role = roleKeyOf(item);
    for (const other of items.slice(0, 500)) {
      if (other.id === item.id) continue;
      const pair = [item.id, other.id].sort().join('|');
      if (seen.has(pair)) continue;
      seen.add(pair);
      if (roleKeyOf(other) !== role) continue;
      push({
        from: `garment:${item.id}`,
        to: `garment:${other.id}`,
        kind: 'similar_to',
        weight: 0.7,
        evidence: `Same wardrobe role ${role}`,
      });
      push({
        from: `garment:${item.id}`,
        to: `garment:${other.id}`,
        kind: 'duplicate_role',
        weight: 0.6,
        evidence: `Duplicate role ${role}`,
      });
      void byId;
    }
  }
  return { nodes: nodesOf(items, memory), edges };
}

/** BFS neighborhood capped at depth 2 and 60 nodes. */
export function neighborhood(
  graph: WardrobeGraph,
  start: string,
  depth = 2,
): { nodes: string[]; edges: WardrobeGraphEdge[] } {
  const adj = new Map<string, WardrobeGraphEdge[]>();
  for (const e of graph.edges) {
    const list = adj.get(e.from) || [];
    list.push(e);
    adj.set(e.from, list);
  }
  const visited = new Set<string>([start]);
  let frontier = [start];
  const edges: WardrobeGraphEdge[] = [];
  for (let d = 0; d < depth && frontier.length; d++) {
    const next: string[] = [];
    for (const node of frontier) {
      for (const e of adj.get(node) || []) {
        edges.push(e);
        if (!visited.has(e.to) && visited.size < 60) {
          visited.add(e.to);
          next.push(e.to);
        }
      }
    }
    frontier = next;
  }
  return { nodes: [...visited], edges: edges.slice(0, 200) };
}

/** Sanitized analytics: counts only, no ids/photos/body data (§39). */
export function sanitizedGraphStats(graph: WardrobeGraph): Record<string, number> {
  const kinds: Record<string, number> = {};
  for (const e of graph.edges) kinds[e.kind] = (kinds[e.kind] || 0) + 1;
  return { nodes: graph.nodes.length, edges: graph.edges.length, ...kinds };
}
