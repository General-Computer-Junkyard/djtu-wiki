# container

Rust 中，容器是存储和组织多个值的数据结构。Rust 提供了多种容器类型，每种容器都有其特定的用途和特点。以下是 Rust 中常用的容器：

1. **Vector (`Vec<T>`)**
   * 是一个可增长的数组，用于存储同一类型的元素。
   * 有助于动态地添加或删除元素。
2. **String (`String`)**
   * 是一个 UTF-8 编码的字符串类型，可以动态增长。
   * 不仅仅是一个字符数组，因为 UTF-8 编码的字符可以占用多个字节。
3. **Hash Map (`HashMap<K, V>`)**
   * 是一个基于键的值存储系统。
   * 允许你存储值并使用唯一的键来快速检索它们。
4. **HashSet (`HashSet<T>`)**
   * 是一个集合，其中每个元素都是唯一的。
   * 有助于快速检查一个值是否存在于集合中，因为它使用哈希进行存储。
5. **LinkedList (`LinkedList<T>`)**
   * 是一个双向链表。
   * 允许快速的前后插入，但随机访问较慢。
6. **BinaryHeap (`BinaryHeap<T>`)**
   * 是一个优先队列。
   * 允许你保持元素的排序，以便始终能够快速访问最大或最小的元素。

每种容器都有其独特的性质和使用场景。选择哪种容器取决于你的具体需求，例如，你需要快速随机访问、快速插入、键值对存储等。

**Vec< T>**

`Vec<T>` 是 Rust 中的一个动态数组（或称为向量），它可以在运行时增长或缩小。`Vec<T>` 是一个泛型容器，其中 `T` 表示它将存储的元素的类型。以下是关于 `Vec<T>` 的一些基本信息和使用示例：

**创建：**

1.  **使用 `vec!` 宏创建向量**:

    ```
    let v = vec![1, 2, 3, 4, 5];
    ```
2.  **使用 `Vec::new()` 方法创建一个空向量**:

    ```
    let mut v: Vec<i32> = Vec::new();
    ```

**添加元素：**

*   使用 `push` 方法向向量的尾部添加元素

    ```
    let mut v = vec![1, 2];
    v.push(3);
    ```

**访问元素：**

1.  **使用索引直接访问**:

    ```
    let v = vec![1, 2, 3];
    let third = v[2];  // third 的值为 3
    ```
2.  **使用 `get` 方法安全地访问元素（返回 `Option<&T>`）**:

    ```
    let v = vec![1, 2, 3];
    let third = v.get(2);  // third 的值为 Some(&3)
    ```

**遍历元素：**

*   使用 `for` 循环遍历向量中的所有元素

    ```
    let v = vec![1, 2, 3];
    for item in &v {
        println!("{}", item);
    }
    ```

**移除元素：**

*   使用 `pop` 方法从向量的尾部移除元素

    ```
    let mut v = vec![1, 2, 3];
    v.pop();  // v 现在为 [1, 2]
    ```

**大小和容量：**

* 使用 `len` 方法获取向量的长度。
* 使用 `capacity` 方法获取向量的容量。
* 使用 `shrink_to_fit` 方法将向量的容量减少到与其长度相同。

**其他特点：**

1. **自动增长**：当向量的大小超过其容量时，它会自动分配新的内存并将旧元素移至新位置。
2. **连续内存**：向量的元素在内存中是连续存储的，这使得它对于随机访问非常快。
3. **拥有其数据**：当 `Vec<T>` 被丢弃时，它的所有元素也都会被丢弃。

**VecDeque< T>**

VecDeque< T> 是 Rust 中的双端队列 (double-ended queue)，它允许在队列的两端高效地添加和移除元素。这与 Vec< T> 不同，Vec< T> 优化了从尾部添加和移除元素，但从头部进行操作可能会非常慢。而 VecDeque< T> 则提供了均衡的性能。

以下是 VecDeque< T> 的一些基本用法和特性：

**创建**使用 VecDeque::new() 创建一个空的 VecDeque\<T>：

```
let mut deque: VecDeque<i32> = VecDeque::new();
```

**添加元素** 使用 push\_back 在队列尾部添加元素：

```
deque.push_back(1);
```

使用 push\_front 在队列头部添加元素：

```
deque.push_front(0);
```

**移除元素** 使用 pop\_back 从队列尾部移除元素：

```
deque.pop_back();  // 返回 Some(1)
```

**使用 pop\_front 从队列头部移除元素：**

```
deque.pop_front();  // 返回 Some(0)
```

**其他操作** front 和 back 可以获取队列的首元素和尾元素，但不移除它们。 len 返回队列中的元素数量。 is\_empty 检查队列是否为空。 clear 清空队列的所有元素。

特性 连续内存：VecDeque\<T> 内部使用一个环形缓冲区。这意味着物理内存可能是连续的，但逻辑上它可以被视为两部分。 自动增长：当 VecDeque\<T> 的容量不足以容纳新元素时，它会自动增长。 VecDeque\<T> 在需要从头部和尾部都进行添加和移除操作时是非常有用的，比如在实现某些算法（如 BFS 广度优先搜索）或数据结构（如滑动窗口）时。

**LinkedLisk< T>**

在 Rust 的标准库中，没有直接名为 `LinkedList<T>` 的数据结构。但是，有一个名为 `std::collections::LinkedList<T>` 的双向链表实现。这是一个基于节点的双向链表，允许 O(1) 的插入和删除操作。

下面是 `std::collections::LinkedList<T>` 的一些基本用法：

**创建**

*   使用 `LinkedList::new()`创建一个空的 `LinkedList<T>`

    ```
    use std::collections::LinkedList;
    ​
    let mut list: LinkedList<i32> = LinkedList::new();
    ```

**添加元素**

*   使用 `push_back` 在链表尾部添加元素：

    ```
    list.push_back(1);
    ```
*   使用 `push_front` 在链表头部添加元素

    ```
    list.push_front(0);
    ```

**移除元素**

*   使用 `pop_back` 从链表尾部移除元素：

    ```
    rustCopy code
    list.pop_back();  // 返回 Some(1)
    ```
*   使用 `pop_front` 从链表头部移除元素：

    ```
    rustCopy code
    list.pop_front();  // 返回 Some(0)
    ```

**其他操作**

* `front` 和 `back` 可以获取链表的首元素和尾元素，但不移除它们。
* `len` 返回链表中的元素数量。
* `is_empty` 检查链表是否为空。
* `clear` 清空链表的所有元素。

**特性**

* **节点基础**：`LinkedList<T>` 是基于节点的，每个节点都在堆上分配。
* **双向**：它是一个双向链表，这意味着每个节点都有一个到前一个和后一个节点的引用。

**HashMap\<K,V>/BTreeMap\<K,V>**

`HashMap<K, V>` 和 `BTreeMap<K, V>` 都是 Rust 中的关联数组（或称为字典）实现，它们允许用户存储键值对，并根据键快速查找值。但它们的内部实现和特性有所不同。

**`HashMap<K, V>`**

**特点**:

* 基于哈希表实现。
* 插入、删除和查找操作通常是 O(1)。
* 键的顺序是不确定的，每次迭代可能都会变化。

**用法**:

```
use std::collections::HashMap;
​
let mut scores = HashMap::new();
​
scores.insert("Blue", 10);
scores.insert("Red", 50);
​
let team_name = String::from("Blue");
let score = scores.get(&team_name);
```

**`BTreeMap<K, V>`**

**特点**:

* 基于平衡二叉搜索树（B-Tree）实现。
* 插入、删除和查找操作都是 O(log n)。
* 键是有序的，这意味着你可以对它们进行排序的迭代。

**用法**:

```
use std::collections::BTreeMap;
​
let mut map = BTreeMap::new();
​
map.insert(3, "c");
map.insert(2, "b");
map.insert(1, "a");
​
for (key, value) in &map {
    println!("{}: {}", key, value);
}
```

**何时选择哪一个？**

* **性能**: 对于大多数用途，`HashMap` 由于其常数时间的操作通常更快。
* **键的顺序**: 如果你需要有序的键，例如范围查询或顺序迭代，那么 `BTreeMap` 是一个好选择。
* **确定性**: `HashMap` 默认使用一个随机的哈希种子，这意味着它的迭代顺序在每次运行时都可能不同。如果你需要更加确定性的行为（例如，为了测试），`BTreeMap` 提供了一个固定的顺序。

总之，选择 `HashMap` 还是 `BTreeMap` 取决于你的具体需求和所面临的性能考虑。在许多情况下，`HashMap` 都是一个很好的默认选择，但如果你需要有序或确定性的行为，那么 `BTreeMap` 可能更合适。

**创建新的哈希表**

问：什么是哈希表？

是一种存储键值对的数据结构，它支持近似常数时间的插入、删除和查找操作。哈希表的工作原理基于一个简单的原则：使用一个哈希函数将键转换为一个数组索引，然后在该索引处存储相应的值。

以下是哈希表的一些关键特点和原理：

1. **哈希函数**：哈希函数负责将键转换为数组索引。理想情况下，哈希函数应该均匀地分布键，以避免太多的键映射到同一索引（这称为哈希冲突）。
2. **冲突解决**：当两个或多个键的哈希值相同时，会发生哈希冲突。常见的冲突解决策略有开放寻址和链地址法。
   * **开放寻址**：当冲突发生时，寻找下一个空闲的槽位存储键值对。
   * **链地址法**：每个槽位包含一个链表或另一种数据结构，用于存储所有哈希到该槽位的键值对。
3. **动态调整大小**：为了维持操作的高效性，哈希表可能需要根据其大小和填充因子（即存储的键值对数与总槽数的比例）动态调整其大小。
4. **性能**：在没有冲突的情况下，哈希表的插入、删除和查找操作通常是常数时间的。但是，当冲突增加时，性能可能下降。
5. **应用**：哈希表在很多应用中都非常有用，例如数据库、缓存、查找表等。

在 Rust 中，哈希表由 `std::collections::HashMap` 类型表示。它提供了创建、修改和查询哈希表的方法。例如，你可以使用 `insert` 方法添加元素，使用 `get` 方法根据键查找值，或使用 `remove` 方法删除键值对。

在 Rust 中，你可以使用 `HashMap` 来创建一个哈希表。以下是如何创建新的哈希表的几种方法：

**使用 `new` 方法**

创建一个空的 `HashMap`：

```
use std::collections::HashMap;
​
let mut map = HashMap::new();
```

**使用 `insert` 方法添加键值对**

创建了一个空的 `HashMap` 后，你可以使用 `insert` 方法向其添加键值对：

```
map.insert("name", "Alice");
map.insert("age", "30");
```

**使用 `collect` 方法从迭代器创建**

你可以使用一个迭代器（例如一个由元组组成的数组或向量的迭代器）和 `collect` 方法来创建一个 `HashMap`：

```
let data = [("name", "Alice"), ("age", "30")];
let map: HashMap<_, _> = data.iter().cloned().collect();
```

**使用 `HashMap::with_capacity`**

如果你预先知道哈希表的大小，可以使用 `with_capacity` 方法为其预分配空间，这可以避免随后的重新分配：

```
let mut map = HashMap::with_capacity(10);
map.insert("name", "Alice");
```

一旦你有了一个 `HashMap`，你就可以使用其提供的各种方法来插入、删除或查找键值对，以及执行其他操作。

**访问哈希表的元素**

1.  **使用 `get` 方法**:

    * `get` 方法用于根据给定的键查找值。
    * 它返回一个 `Option`，其中 `Some(&value)` 表示找到了值，`None` 表示未找到。

    ```
    use std::collections::HashMap;
    ​
    let mut scores = HashMap::new();
    scores.insert(String::from("Blue"), 10);
    scores.insert(String::from("Yellow"), 50);
    ​
    let team_name = String::from("Blue");
    let score = scores.get(&team_name);
    match score {
        Some(&n) => println!("Score for {}: {}", team_name, n),
        None => println!("No score for team {}", team_name),
    }
    ```
2.  **直接使用键访问**:

    * 使用`[key]`语法直接访问元素。但这在键不存在时会引发恐慌。
    * 使用`entry(key)`与`or_insert(value)`结合可以更安全地访问和插入。

    ```
    let mut count = HashMap::new();
    count.entry(String::from("Blue")).or_insert(10);
    ```
3.  **遍历HashMap**:

    * 使用`for`循环与`iter`方法结合可以遍历哈希表的所有键值对。

    ```
    for (key, value) in &scores {
        println!("{}: {}", key, value);
    }
    ```
4.  **更新值**:

    * 可以使用`insert`方法覆盖旧值。
    * 使用`entry(key).or_insert(value)`结合可以只在键不存在时插入。

    ```
    *scores.entry(String::from("Blue")).or_insert(0) += 10;
    ```
5.  **删除元素**:

    * 使用`remove`方法删除键和其关联的值。

    ```
    scores.remove(&String::from("Blue"));
    ```

**迭代哈希表**

在 Rust 中，可以使用 `for` 循环遍历 `HashMap`（哈希表）的内容。以下是如何迭代哈希表的基本方法：

1.  **遍历所有的键值对**:

    ```
    use std::collections::HashMap;
    ​
    let mut map = HashMap::new();
    map.insert("key1", "value1");
    map.insert("key2", "value2");
    ​
    for (key, value) in &map {
        println!("Key: {}, Value: {}", key, value);
    }
    ```
2.  **只遍历所有的键**:

    ```
    for key in map.keys() {
        println!("Key: {}", key);
    }
    ```
3.  **只遍历所有的值**:

    ```
    for value in map.values() {
        println!("Value: {}", value);
    }
    ```
4.  **可变引用遍历**: 如果你想在迭代过程中修改哈希表的值，可以使用可变引用来遍历：

    ```
    for value in map.values_mut() {
        // 这里可以修改 value
    }
    ```

注意：由于 `HashMap` 是基于哈希的，所以遍历的顺序并不保证与插入的顺序相同。

**哈希表和所有权**

在 Rust 中，哈希表（`HashMap`）和所有权的概念紧密相连。以下是一些与 `HashMap` 和所有权相关的重要点：

1.  **所有权转移**: 当将值插入到 `HashMap` 中时，这些值的所有权会被转移给哈希表。这意味着原始值将不再可用。

    ```
    use std::collections::HashMap;
    ​
    let key = String::from("name");
    let value = String::from("Alice");
    ​
    let mut map = HashMap::new();
    map.insert(key, value);
    ​
    // 此时 key 和 value 都不再可用，因为它们的所有权已经转移到了 map 中。
    ```
2.  **引用作为键或值**: 可以使用引用作为键或值，但需要确保这些引用在哈希表的整个生命周期中都有效。这通常涉及到生命周期注解。

    ```
    let key = String::from("name");
    let value = String::from("Alice");
    ​
    let mut map: HashMap<&str, &str> = HashMap::new();
    map.insert(&key, &value);
    ```
3. **获取值**: 使用 `get` 方法从 `HashMap` 获取值时，获得的是一个引用。如果需要拥有值的所有权，需要使用 `remove` 方法。
4.  **更新值**: 更新 `HashMap` 中的值不需要取出值，修改它然后再放回去。你可以直接使用如 `entry` 和 `or_insert` 这样的方法。

    ```
    rustCopy code
    *map.entry(String::from("name")).or_insert(String::from("Bob")) += " Smith";
    ```
5. **删除值**: 使用 `remove` 方法从 `HashMap` 中删除键值对会将值返回给你，这样你就获得了这个值的所有权。

**HashSet< T>/BTreeSet< T>**

`HashSet<T>` 和 `BTreeSet<T>` 都是 Rust 中的集合（set）数据结构，它们提供了一组不重复的元素集。这两种集合在某些方面类似，但它们的内部实现和某些特性是不同的。

1.  **HashSet< T>**:

    * 基于哈希表实现。
    * 插入、删除和查找的平均时间复杂度为 O(1)。
    * 不保证元素的顺序。
    * 快速的成员检查。

    ```
    use std::collections::HashSet;
    ​
    let mut fruits = HashSet::new();
    fruits.insert("apple");
    fruits.insert("banana");
    fruits.insert("orange");
    ```
2.  **BTreeSet\<T>**:

    * 基于平衡二叉树实现。
    * 插入、删除和查找的时间复杂度为 O(log n)。
    * 元素总是有序的。
    * 支持有序集合的操作，如找到最小或最大的元素。

    ```
    rustCopy codeuse std::collections::BTreeSet;
    ​
    let mut numbers = BTreeSet::new();
    numbers.insert(3);
    numbers.insert(1);
    numbers.insert(4);
    ```

**比较**:

* **性能**: 对于大多数使用情况，`HashSet` 的性能可能更好，因为它提供了近似常数时间的操作。但是，如果需要有序的集合或范围查询，`BTreeSet` 可能是更好的选择。
* **功能**: `BTreeSet` 提供了一些有序集合特有的功能，如 `range` 方法，它允许对一定范围内的元素进行迭代。
* **内存**: 由于 `BTreeSet` 的树形结构，它可能会使用比 `HashSet` 更多的内存。

\