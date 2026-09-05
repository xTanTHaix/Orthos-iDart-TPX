use parking_lot::RwLock;

/// Adaptive Radix Tree (ART) implementation.
/// Implements cache-line-padded node structures (Node4, Node16, Node48, Node256).
/// Supports arbitrary keys including keys that are prefixes of other keys, empty keys, and binary keys.

#[repr(align(64))]
enum NodeBody<V> {
    Node4 {
        keys: [u8; 4],
        children: [Option<Box<ArtNode<V>>>; 4],
        num_children: u8,
        val: Option<(Vec<u8>, V)>,
    },
    Node16 {
        keys: [u8; 16],
        children: [Option<Box<ArtNode<V>>>; 16],
        num_children: u8,
        val: Option<(Vec<u8>, V)>,
    },
    Node48 {
        index: [u8; 256],
        children: [Option<Box<ArtNode<V>>>; 48],
        num_children: u8,
        val: Option<(Vec<u8>, V)>,
    },
    Node256 {
        children: [Option<Box<ArtNode<V>>>; 256],
        num_children: u16,
        val: Option<(Vec<u8>, V)>,
    },
    Leaf {
        key: Vec<u8>,
        value: V,
    },
}

struct ArtNode<V> {
    body: NodeBody<V>,
}

impl<V: Clone> ArtNode<V> {
    fn new_leaf(key: Vec<u8>, value: V) -> Self {
        Self {
            body: NodeBody::Leaf { key, value },
        }
    }

    fn new_node4() -> Self {
        Self {
            body: NodeBody::Node4 {
                keys: [0; 4],
                children: [None, None, None, None],
                num_children: 0,
                val: None,
            },
        }
    }

    fn get_node_val(&self) -> Option<&(Vec<u8>, V)> {
        match &self.body {
            NodeBody::Node4 { val, .. }
            | NodeBody::Node16 { val, .. }
            | NodeBody::Node48 { val, .. }
            | NodeBody::Node256 { val, .. } => val.as_ref(),
            NodeBody::Leaf { .. } => None,
        }
    }

    #[allow(dead_code)]
    fn set_node_val(&mut self, key: Vec<u8>, value: V) -> Option<V> {
        match &mut self.body {
            NodeBody::Node4 { val, .. }
            | NodeBody::Node16 { val, .. }
            | NodeBody::Node48 { val, .. }
            | NodeBody::Node256 { val, .. } => {
                let old = val.take().map(|(_, v)| v);
                *val = Some((key, value));
                old
            }
            NodeBody::Leaf { .. } => None,
        }
    }

    #[allow(dead_code)]
    fn take_node_val(&mut self) -> Option<V> {
        match &mut self.body {
            NodeBody::Node4 { val, .. }
            | NodeBody::Node16 { val, .. }
            | NodeBody::Node48 { val, .. }
            | NodeBody::Node256 { val, .. } => val.take().map(|(_, v)| v),
            NodeBody::Leaf { .. } => None,
        }
    }
}

pub struct ArtTree<V> {
    root: RwLock<Option<Box<ArtNode<V>>>>,
}

impl<V: Clone> Default for ArtTree<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Clone> ArtTree<V> {
    pub fn new() -> Self {
        Self {
            root: RwLock::new(None),
        }
    }

    pub fn insert(&self, key: &[u8], value: V) -> Option<V> {
        let mut root_guard = self.root.write();
        if root_guard.is_none() {
            *root_guard = Some(Box::new(ArtNode::new_leaf(key.to_vec(), value)));
            return None;
        }

        Self::insert_rec(root_guard.as_mut().unwrap(), key, value, 0)
    }

    fn insert_rec(node: &mut Box<ArtNode<V>>, key: &[u8], value: V, depth: usize) -> Option<V> {
        match &mut node.body {
            NodeBody::Leaf {
                key: leaf_key,
                value: leaf_val,
            } => {
                if leaf_key.as_slice() == key {
                    let old = leaf_val.clone();
                    *leaf_val = value;
                    return Some(old);
                }

                let old_key = leaf_key.clone();
                let old_val = leaf_val.clone();

                let old_byte = old_key.get(depth).copied();
                let new_byte = key.get(depth).copied();

                if old_byte == new_byte && old_byte.is_some() {
                    // Continue sharing prefix: recurse down
                    let mut child = Box::new(ArtNode::new_leaf(old_key, old_val));
                    Self::insert_rec(&mut child, key, value, depth + 1);

                    let mut intermediate = Box::new(ArtNode::new_node4());
                    Self::add_child(&mut intermediate, old_byte.unwrap(), child);
                    *node = intermediate;
                    return None;
                }

                let mut new_node = Box::new(ArtNode::new_node4());
                if let Some(ob) = old_byte {
                    Self::add_child(
                        &mut new_node,
                        ob,
                        Box::new(ArtNode::new_leaf(old_key, old_val)),
                    );
                } else {
                    new_node.set_node_val(old_key, old_val);
                }

                if let Some(nb) = new_byte {
                    Self::add_child(
                        &mut new_node,
                        nb,
                        Box::new(ArtNode::new_leaf(key.to_vec(), value)),
                    );
                } else {
                    new_node.set_node_val(key.to_vec(), value);
                }

                *node = new_node;
                None
            }
            NodeBody::Node4 {
                keys,
                children,
                num_children,
                val,
            } => {
                let byte = match key.get(depth) {
                    Some(&b) => b,
                    None => {
                        let old = val.take().map(|(_, v)| v);
                        *val = Some((key.to_vec(), value));
                        return old;
                    }
                };

                for i in 0..*num_children as usize {
                    if keys[i] == byte {
                        return Self::insert_rec(
                            children[i].as_mut().unwrap(),
                            key,
                            value,
                            depth + 1,
                        );
                    }
                }

                if (*num_children as usize) < 4 {
                    let idx = *num_children as usize;
                    keys[idx] = byte;
                    children[idx] = Some(Box::new(ArtNode::new_leaf(key.to_vec(), value)));
                    *num_children += 1;
                    None
                } else {
                    // Grow to Node16
                    let mut n16_keys = [0u8; 16];
                    let mut n16_children: [Option<Box<ArtNode<V>>>; 16] =
                        std::array::from_fn(|_| None);
                    for i in 0..4 {
                        n16_keys[i] = keys[i];
                        n16_children[i] = children[i].take();
                    }
                    n16_keys[4] = byte;
                    n16_children[4] = Some(Box::new(ArtNode::new_leaf(key.to_vec(), value)));

                    node.body = NodeBody::Node16 {
                        keys: n16_keys,
                        children: n16_children,
                        num_children: 5,
                        val: val.take(),
                    };
                    None
                }
            }
            NodeBody::Node16 {
                keys,
                children,
                num_children,
                val,
            } => {
                let byte = match key.get(depth) {
                    Some(&b) => b,
                    None => {
                        let old = val.take().map(|(_, v)| v);
                        *val = Some((key.to_vec(), value));
                        return old;
                    }
                };

                for i in 0..*num_children as usize {
                    if keys[i] == byte {
                        return Self::insert_rec(
                            children[i].as_mut().unwrap(),
                            key,
                            value,
                            depth + 1,
                        );
                    }
                }

                if (*num_children as usize) < 16 {
                    let idx = *num_children as usize;
                    keys[idx] = byte;
                    children[idx] = Some(Box::new(ArtNode::new_leaf(key.to_vec(), value)));
                    *num_children += 1;
                    None
                } else {
                    // Grow to Node48
                    let mut index = [0xFFu8; 256];
                    let mut n48_children: [Option<Box<ArtNode<V>>>; 48] =
                        std::array::from_fn(|_| None);
                    for i in 0..16 {
                        index[keys[i] as usize] = i as u8;
                        n48_children[i] = children[i].take();
                    }
                    index[byte as usize] = 16;
                    n48_children[16] = Some(Box::new(ArtNode::new_leaf(key.to_vec(), value)));

                    node.body = NodeBody::Node48 {
                        index,
                        children: n48_children,
                        num_children: 17,
                        val: val.take(),
                    };
                    None
                }
            }
            NodeBody::Node48 {
                index,
                children,
                num_children,
                val,
            } => {
                let byte = match key.get(depth) {
                    Some(&b) => b,
                    None => {
                        let old = val.take().map(|(_, v)| v);
                        *val = Some((key.to_vec(), value));
                        return old;
                    }
                };

                let slot = index[byte as usize];
                if slot != 0xFF {
                    return Self::insert_rec(
                        children[slot as usize].as_mut().unwrap(),
                        key,
                        value,
                        depth + 1,
                    );
                }

                if (*num_children as usize) < 48 {
                    let slot = *num_children as usize;
                    index[byte as usize] = slot as u8;
                    children[slot] = Some(Box::new(ArtNode::new_leaf(key.to_vec(), value)));
                    *num_children += 1;
                    None
                } else {
                    // Grow to Node256
                    let mut n256_children: [Option<Box<ArtNode<V>>>; 256] =
                        std::array::from_fn(|_| None);
                    for b in 0..256 {
                        let s = index[b];
                        if s != 0xFF {
                            n256_children[b] = children[s as usize].take();
                        }
                    }
                    n256_children[byte as usize] =
                        Some(Box::new(ArtNode::new_leaf(key.to_vec(), value)));

                    node.body = NodeBody::Node256 {
                        children: n256_children,
                        num_children: 49,
                        val: val.take(),
                    };
                    None
                }
            }
            NodeBody::Node256 {
                children,
                num_children,
                val,
            } => {
                let byte = match key.get(depth) {
                    Some(&b) => b,
                    None => {
                        let old = val.take().map(|(_, v)| v);
                        *val = Some((key.to_vec(), value));
                        return old;
                    }
                };

                if let Some(child) = children[byte as usize].as_mut() {
                    Self::insert_rec(child, key, value, depth + 1)
                } else {
                    children[byte as usize] =
                        Some(Box::new(ArtNode::new_leaf(key.to_vec(), value)));
                    *num_children += 1;
                    None
                }
            }
        }
    }

    fn add_child(node: &mut Box<ArtNode<V>>, byte: u8, child: Box<ArtNode<V>>) {
        if let NodeBody::Node4 {
            keys,
            children,
            num_children,
            ..
        } = &mut node.body
        {
            let idx = *num_children as usize;
            if idx < 4 {
                keys[idx] = byte;
                children[idx] = Some(child);
                *num_children += 1;
            }
        }
    }

    pub fn get(&self, key: &[u8]) -> Option<V> {
        let root_guard = self.root.read();
        let mut current = root_guard.as_ref()?;
        let mut depth = 0;

        loop {
            match &current.body {
                NodeBody::Leaf {
                    key: leaf_key,
                    value,
                } => {
                    if leaf_key.as_slice() == key {
                        return Some(value.clone());
                    } else {
                        return None;
                    }
                }
                NodeBody::Node4 {
                    keys,
                    children,
                    num_children,
                    val,
                } => {
                    let byte = match key.get(depth) {
                        Some(&b) => b,
                        None => return val.as_ref().map(|(_, v)| v.clone()),
                    };
                    let mut found = false;
                    for i in 0..*num_children as usize {
                        if keys[i] == byte {
                            current = children[i].as_ref()?;
                            depth += 1;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return None;
                    }
                }
                NodeBody::Node16 {
                    keys,
                    children,
                    num_children,
                    val,
                } => {
                    let byte = match key.get(depth) {
                        Some(&b) => b,
                        None => return val.as_ref().map(|(_, v)| v.clone()),
                    };
                    let mut found = false;
                    for i in 0..*num_children as usize {
                        if keys[i] == byte {
                            current = children[i].as_ref()?;
                            depth += 1;
                            found = true;
                            break;
                        }
                    }
                    if !found {
                        return None;
                    }
                }
                NodeBody::Node48 {
                    index,
                    children,
                    val,
                    ..
                } => {
                    let byte = match key.get(depth) {
                        Some(&b) => b,
                        None => return val.as_ref().map(|(_, v)| v.clone()),
                    };
                    let slot = index[byte as usize];
                    if slot == 0xFF {
                        return None;
                    }
                    current = children[slot as usize].as_ref()?;
                    depth += 1;
                }
                NodeBody::Node256 { children, val, .. } => {
                    let byte = match key.get(depth) {
                        Some(&b) => b,
                        None => return val.as_ref().map(|(_, v)| v.clone()),
                    };
                    current = children[byte as usize].as_ref()?;
                    depth += 1;
                }
            }
        }
    }

    pub fn remove(&self, key: &[u8]) -> Option<V> {
        let mut root_guard = self.root.write();
        let current = root_guard.as_mut()?;

        if let NodeBody::Leaf {
            key: leaf_key,
            value,
        } = &current.body
        {
            if leaf_key.as_slice() == key {
                let old = value.clone();
                *root_guard = None;
                return Some(old);
            }
            return None;
        }

        Self::remove_rec(current, key, 0)
    }

    fn remove_rec(node: &mut Box<ArtNode<V>>, key: &[u8], depth: usize) -> Option<V> {
        match &mut node.body {
            NodeBody::Leaf {
                key: leaf_key,
                value,
            } => {
                if leaf_key.as_slice() == key {
                    Some(value.clone())
                } else {
                    None
                }
            }
            NodeBody::Node4 {
                keys,
                children,
                num_children,
                val,
            } => {
                let byte = match key.get(depth) {
                    Some(&b) => b,
                    None => return val.take().map(|(_, v)| v),
                };
                for i in 0..*num_children as usize {
                    if keys[i] == byte {
                        if let Some(child) = &mut children[i] {
                            if let NodeBody::Leaf {
                                key: leaf_key,
                                value,
                            } = &child.body
                            {
                                if leaf_key.as_slice() == key {
                                    let old = value.clone();
                                    children[i] = None;
                                    for j in i..(*num_children as usize - 1) {
                                        keys[j] = keys[j + 1];
                                        children[j] = children[j + 1].take();
                                    }
                                    *num_children -= 1;
                                    return Some(old);
                                }
                            }
                            return Self::remove_rec(child, key, depth + 1);
                        }
                    }
                }
                None
            }
            NodeBody::Node16 {
                keys,
                children,
                num_children,
                val,
            } => {
                let byte = match key.get(depth) {
                    Some(&b) => b,
                    None => return val.take().map(|(_, v)| v),
                };
                for i in 0..*num_children as usize {
                    if keys[i] == byte {
                        if let Some(child) = &mut children[i] {
                            if let NodeBody::Leaf {
                                key: leaf_key,
                                value,
                            } = &child.body
                            {
                                if leaf_key.as_slice() == key {
                                    let old = value.clone();
                                    children[i] = None;
                                    for j in i..(*num_children as usize - 1) {
                                        keys[j] = keys[j + 1];
                                        children[j] = children[j + 1].take();
                                    }
                                    *num_children -= 1;
                                    return Some(old);
                                }
                            }
                            return Self::remove_rec(child, key, depth + 1);
                        }
                    }
                }
                None
            }
            NodeBody::Node48 {
                index,
                children,
                num_children,
                val,
            } => {
                let byte = match key.get(depth) {
                    Some(&b) => b,
                    None => return val.take().map(|(_, v)| v),
                };
                let slot = index[byte as usize];
                if slot == 0xFF {
                    return None;
                }
                if let Some(child) = &mut children[slot as usize] {
                    if let NodeBody::Leaf {
                        key: leaf_key,
                        value,
                    } = &child.body
                    {
                        if leaf_key.as_slice() == key {
                            let old = value.clone();
                            children[slot as usize] = None;
                            index[byte as usize] = 0xFF;
                            *num_children -= 1;
                            return Some(old);
                        }
                    }
                    return Self::remove_rec(child, key, depth + 1);
                }
                None
            }
            NodeBody::Node256 {
                children,
                num_children,
                val,
            } => {
                let byte = match key.get(depth) {
                    Some(&b) => b,
                    None => return val.take().map(|(_, v)| v),
                };
                if let Some(child) = &mut children[byte as usize] {
                    if let NodeBody::Leaf {
                        key: leaf_key,
                        value,
                    } = &child.body
                    {
                        if leaf_key.as_slice() == key {
                            let old = value.clone();
                            children[byte as usize] = None;
                            *num_children -= 1;
                            return Some(old);
                        }
                    }
                    return Self::remove_rec(child, key, depth + 1);
                }
                None
            }
        }
    }

    pub fn get_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, V)> {
        let root_guard = self.root.read();
        let mut results = Vec::new();
        if let Some(root) = root_guard.as_ref() {
            Self::collect_prefix(root, prefix, 0, &mut results);
        }
        results
    }

    fn collect_prefix(
        node: &Box<ArtNode<V>>,
        prefix: &[u8],
        depth: usize,
        results: &mut Vec<(Vec<u8>, V)>,
    ) {
        if depth >= prefix.len() {
            Self::collect_all(node, results);
            return;
        }

        if let Some((k, v)) = node.get_node_val() {
            if k.starts_with(prefix) {
                results.push((k.clone(), v.clone()));
            }
        }

        let target_byte = prefix[depth];

        match &node.body {
            NodeBody::Leaf { key, value } => {
                if key.starts_with(prefix) {
                    results.push((key.clone(), value.clone()));
                }
            }
            NodeBody::Node4 {
                keys,
                children,
                num_children,
                ..
            } => {
                for i in 0..*num_children as usize {
                    if keys[i] == target_byte {
                        if let Some(child) = &children[i] {
                            Self::collect_prefix(child, prefix, depth + 1, results);
                        }
                    }
                }
            }
            NodeBody::Node16 {
                keys,
                children,
                num_children,
                ..
            } => {
                for i in 0..*num_children as usize {
                    if keys[i] == target_byte {
                        if let Some(child) = &children[i] {
                            Self::collect_prefix(child, prefix, depth + 1, results);
                        }
                    }
                }
            }
            NodeBody::Node48 {
                index, children, ..
            } => {
                let slot = index[target_byte as usize];
                if slot != 0xFF {
                    if let Some(child) = &children[slot as usize] {
                        Self::collect_prefix(child, prefix, depth + 1, results);
                    }
                }
            }
            NodeBody::Node256 { children, .. } => {
                if let Some(child) = &children[target_byte as usize] {
                    Self::collect_prefix(child, prefix, depth + 1, results);
                }
            }
        }
    }

    fn collect_all(node: &Box<ArtNode<V>>, results: &mut Vec<(Vec<u8>, V)>) {
        if let Some((k, v)) = node.get_node_val() {
            results.push((k.clone(), v.clone()));
        }

        match &node.body {
            NodeBody::Leaf { key, value } => {
                results.push((key.clone(), value.clone()));
            }
            NodeBody::Node4 {
                children,
                num_children,
                ..
            } => {
                for i in 0..*num_children as usize {
                    if let Some(child) = &children[i] {
                        Self::collect_all(child, results);
                    }
                }
            }
            NodeBody::Node16 {
                children,
                num_children,
                ..
            } => {
                for i in 0..*num_children as usize {
                    if let Some(child) = &children[i] {
                        Self::collect_all(child, results);
                    }
                }
            }
            NodeBody::Node48 {
                index, children, ..
            } => {
                for &slot in index.iter() {
                    if slot != 0xFF {
                        if let Some(child) = &children[slot as usize] {
                            Self::collect_all(child, results);
                        }
                    }
                }
            }
            NodeBody::Node256 { children, .. } => {
                for child_opt in children.iter() {
                    if let Some(child) = child_opt {
                        Self::collect_all(child, results);
                    }
                }
            }
        }
    }

    pub fn snapshot(&self) -> ArtSnapshot<V> {
        let items = self.get_prefix(b"");
        ArtSnapshot { items }
    }
}

pub struct ArtSnapshot<V> {
    items: Vec<(Vec<u8>, V)>,
}

impl<V: Clone> ArtSnapshot<V> {
    pub fn get(&self, key: &[u8]) -> Option<V> {
        for (k, v) in &self.items {
            if k.as_slice() == key {
                return Some(v.clone());
            }
        }
        None
    }

    pub fn get_prefix(&self, prefix: &[u8]) -> Vec<(Vec<u8>, V)> {
        self.items
            .iter()
            .filter(|(k, _)| k.starts_with(prefix))
            .cloned()
            .collect()
    }
}
