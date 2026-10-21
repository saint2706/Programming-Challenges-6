//! Node storage for path-copying trees, and the version table on top of it.
//!
//! Every tree in this crate is a DAG of immutable nodes in one `Vec`: a node names its children
//! by index, so "copying a path" means pushing a few new entries and pointing them at old ones.
//! Index 0 is [`NIL`], a node that stands for any all-identity subtree, so sparse trees (a count
//! tree over a million possible values that has seen a thousand) cost nothing for what is absent.
//!
//! Dropping an old version needs a way to know which nodes nobody else points at. Two are offered
//! and both work on the same arena:
//!
//! * **Reference counts** (`counting = true`): a `u32` per node counts the parent pointers and
//!   version roots that reach it. Retiring a version releases its root, and every node whose count
//!   falls to zero goes onto a free list for the next allocation. Memory is reclaimed immediately,
//!   at 4 extra bytes per node and an increment per child on every allocation.
//! * **Mark and compact** ([`Store::compact`]): trace everything reachable from the live roots and
//!   copy it, children before parents, into a fresh `Vec`. No per-node overhead, but a pause
//!   proportional to the live data, and it is the only way the arena ever shrinks.

/// Index of a node in the arena.
pub type NodeId = u32;

/// The shared empty subtree: aggregate is the identity, both children are `NIL`.
pub const NIL: NodeId = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Node<A: Copy> {
    pub l: NodeId,
    pub r: NodeId,
    pub agg: A,
}

pub struct Arena<A: Copy> {
    nodes: Vec<Node<A>>,
    rc: Vec<u32>,
    free: Vec<NodeId>,
    scratch: Vec<NodeId>,
    counting: bool,
    live: usize,
}

impl<A: Copy> Arena<A> {
    pub fn new(identity: A, counting: bool) -> Self {
        let nil = Node {
            l: NIL,
            r: NIL,
            agg: identity,
        };
        Self {
            nodes: vec![nil],
            rc: if counting {
                vec![u32::MAX / 2]
            } else {
                Vec::new()
            },
            free: Vec::new(),
            scratch: Vec::new(),
            counting,
            live: 0,
        }
    }

    #[inline]
    pub fn get(&self, id: NodeId) -> &Node<A> {
        &self.nodes[id as usize]
    }

    #[inline]
    pub fn agg(&self, id: NodeId) -> A {
        self.nodes[id as usize].agg
    }

    pub fn counting(&self) -> bool {
        self.counting
    }

    /// New node pointing at `l` and `r`. With counting on, it takes a reference to each child and
    /// starts with none of its own: whoever stores the id (a parent, a version slot) retains it.
    pub fn alloc(&mut self, l: NodeId, r: NodeId, agg: A) -> NodeId {
        let node = Node { l, r, agg };
        let id = match self.free.pop() {
            Some(id) => {
                self.nodes[id as usize] = node;
                id
            }
            None => {
                self.nodes.push(node);
                if self.counting {
                    self.rc.push(0);
                }
                NodeId::try_from(self.nodes.len() - 1)
                    .expect("the arena is full: node ids are 32-bit (about 4.3 billion nodes)")
            }
        };
        if self.counting {
            self.rc[id as usize] = 0;
            self.retain(l);
            self.retain(r);
        }
        self.live += 1;
        id
    }

    #[inline]
    pub fn retain(&mut self, id: NodeId) {
        if self.counting && id != NIL {
            self.rc[id as usize] += 1;
        }
    }

    /// Drops one reference to `id`; frees it, and recursively whatever only it kept alive, when the
    /// count reaches zero. Iterative: a freed chain can be as long as the tree is deep.
    pub fn release(&mut self, id: NodeId) {
        if !self.counting {
            return;
        }
        let mut stack = std::mem::take(&mut self.scratch);
        stack.push(id);
        while let Some(x) = stack.pop() {
            if x == NIL {
                continue;
            }
            let count = &mut self.rc[x as usize];
            debug_assert!(*count > 0, "released node {x} that nothing references");
            *count -= 1;
            if *count == 0 {
                let node = self.nodes[x as usize];
                stack.push(node.l);
                stack.push(node.r);
                self.free.push(x);
                self.live -= 1;
            }
        }
        self.scratch = stack;
    }

    /// Room for `additional` more nodes without reallocating. A growing `Vec` doubles, so without
    /// this an arena can hold up to twice the memory its nodes need.
    pub fn reserve(&mut self, additional: usize) {
        self.nodes.reserve_exact(additional);
        if self.counting {
            self.rc.reserve_exact(additional);
        }
    }

    /// Nodes in use (allocated and not freed), not counting `NIL`.
    pub fn live(&self) -> usize {
        self.live
    }

    /// Slots the arena holds, used or free, not counting `NIL`.
    pub fn slots(&self) -> usize {
        self.nodes.len() - 1
    }

    /// Heap bytes held by the arena's vectors (capacity, not length).
    pub fn bytes(&self) -> usize {
        self.nodes.capacity() * size_of::<Node<A>>()
            + self.rc.capacity() * size_of::<u32>()
            + self.free.capacity() * size_of::<NodeId>()
    }

    /// Mark and compact: keeps exactly what the `roots` reach, children before parents, and
    /// rewrites `roots` to the new indices. Reference counts are rebuilt when counting is on.
    pub fn compact(&mut self, roots: &mut [NodeId]) {
        let mut remap = vec![u32::MAX; self.nodes.len()];
        remap[NIL as usize] = NIL;
        let mut nodes: Vec<Node<A>> = Vec::with_capacity(self.live + 1);
        nodes.push(self.nodes[NIL as usize]);
        // Post-order DFS with an explicit stack: (node, children already pushed).
        let mut stack: Vec<(NodeId, bool)> = Vec::new();
        for root in roots.iter() {
            stack.push((*root, false));
            while let Some((x, expanded)) = stack.pop() {
                if remap[x as usize] != u32::MAX {
                    continue;
                }
                let node = self.nodes[x as usize];
                if expanded {
                    let moved = Node {
                        l: remap[node.l as usize],
                        r: remap[node.r as usize],
                        agg: node.agg,
                    };
                    remap[x as usize] = nodes.len() as NodeId;
                    nodes.push(moved);
                } else {
                    stack.push((x, true));
                    stack.push((node.r, false));
                    stack.push((node.l, false));
                }
            }
        }
        for root in roots.iter_mut() {
            *root = remap[*root as usize];
        }
        if self.counting {
            let mut rc = vec![0u32; nodes.len()];
            rc[NIL as usize] = u32::MAX / 2;
            for node in &nodes[1..] {
                for child in [node.l, node.r] {
                    if child != NIL {
                        rc[child as usize] += 1;
                    }
                }
            }
            for root in roots.iter() {
                if *root != NIL {
                    rc[*root as usize] += 1;
                }
            }
            self.rc = rc;
        }
        // `with_capacity` above was a guess (the pre-collection count); give back what was not used.
        nodes.shrink_to_fit();
        self.live = nodes.len() - 1;
        self.nodes = nodes;
        self.free = Vec::new();
    }
}

/// Identifies a version. Ids are never reused, so a retired version stays retired.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Version(pub u32);

/// An arena plus the table of version roots: what every tree type in this crate owns.
pub struct Store<A: Copy> {
    pub(crate) arena: Arena<A>,
    roots: Vec<Option<NodeId>>,
}

/// Memory and liveness figures for one store.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stats {
    /// Versions created, including retired ones.
    pub versions: usize,
    /// Versions still queryable.
    pub live_versions: usize,
    /// Nodes in use.
    pub live_nodes: usize,
    /// Arena slots, used or free.
    pub slots: usize,
    /// Heap bytes of the arena.
    pub bytes: usize,
}

impl<A: Copy> Store<A> {
    pub fn new(identity: A, counting: bool) -> Self {
        Self {
            arena: Arena::new(identity, counting),
            roots: Vec::new(),
        }
    }

    /// See [`Arena::reserve`].
    pub fn reserve(&mut self, additional_nodes: usize) {
        self.arena.reserve(additional_nodes);
    }

    /// Registers `root` as a new version (taking a reference to it) and returns its id.
    pub fn push_root(&mut self, root: NodeId) -> Version {
        self.arena.retain(root);
        self.roots.push(Some(root));
        Version((self.roots.len() - 1) as u32)
    }

    pub fn root(&self, v: Version) -> NodeId {
        self.roots[v.0 as usize].unwrap_or_else(|| panic!("version {} was retired", v.0))
    }

    pub fn is_live(&self, v: Version) -> bool {
        self.roots.get(v.0 as usize).is_some_and(Option::is_some)
    }

    /// Number of versions ever created.
    pub fn versions(&self) -> usize {
        self.roots.len()
    }

    /// Forgets version `v`. With counting on, the nodes only it reached are freed at once; without,
    /// they stay until [`Store::compact`]. Returns false if it was already retired.
    pub fn retire(&mut self, v: Version) -> bool {
        match self.roots[v.0 as usize].take() {
            Some(root) => {
                self.arena.release(root);
                true
            }
            None => false,
        }
    }

    /// Mark-and-compact the arena down to what the live versions reach.
    pub fn compact(&mut self) {
        let (mut ids, slots): (Vec<NodeId>, Vec<usize>) = self
            .roots
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (r, i)))
            .unzip();
        self.arena.compact(&mut ids);
        for (id, slot) in ids.into_iter().zip(slots) {
            self.roots[slot] = Some(id);
        }
    }

    pub fn stats(&self) -> Stats {
        Stats {
            versions: self.roots.len(),
            live_versions: self.roots.iter().flatten().count(),
            live_nodes: self.arena.live(),
            slots: self.arena.slots(),
            bytes: self.arena.bytes() + self.roots.capacity() * size_of::<Option<NodeId>>(),
        }
    }
}
