# Vector and type conversion

**向量和类型转换**

#### 向量

在 Rust 中，向量（Vector）是一种存储多个值的动态数组或列表。与数组不同，向量的大小是可以变化的，这意味着你可以在运行时向向量中添加或删除元素。

向量是 Rust 的标准库 `std::vec::Vec` 提供的一种数据结构。

以下是关于 Rust 中向量的一些关键点：

1. **动态大小**：与固定大小的数组不同，向量的大小是动态的。这意味着你可以在运行时添加或删除元素。
2. **同一类型的元素**：向量中的所有元素都必须是相同的类型。
3. **内存分配**：向量的元素存储在堆上，这允许它在运行时动态地改变大小。
4. **常用方法**：向量提供了许多方法来操作其内容，例如 `push`（添加元素到向量的末尾）、`pop`（从向量的末尾删除元素）和 `len`（获取向量的长度）等。

以下是一个简单的例子，展示如何在 Rust 中使用向量：

```
// 创建一个新的空向量
let mut numbers = Vec::new();
​
// 使用 push 方法向向量中添加元素
numbers.push(1);
numbers.push(2);
numbers.push(3);
​
// 访问向量中的元素
let first_number = numbers[0]; // 1
​
// 使用 pop 方法从向量的末尾删除元素
numbers.pop(); // 删除了 3
​
// 使用 len 方法获取向量的长度
let length = numbers.len(); // 2
```

* 标准库提供的类型，可以直接使用，Vec是分配在堆上的，可增长的数组

类似c++中的 std::vector, java中的java.util.ArrayList

* \<T>表示泛型，使用时代入实际的类型，例如：元素是i32类型的Vec写作Vec\<i32>
*   使用Vec::new()或vec!宏来创建Vec

    Vec::new()是名字空间的例子，new是定义在Vec结构体中的函数

#### 类型转换

用as进行类型转换（cast）：

```
let x:i32 = 100;
let y:u32 = x as u32;
```

这段代码展示了 Rust 中的类型转换操作。具体来说，它涉及到从一个类型（`i32`，32位有符号整数）转换到另一个类型（`u32`，32位无符号整数）。

以下是代码的逐步解释：

1. `let x: i32 = 100;` 这行代码定义了一个名为 `x` 的变量，类型为 `i32`（32位有符号整数），并将其初始化为 `100`。
2. `let y: u32 = x as u32;` 这行代码进行了几个操作：
   * 它定义了一个名为 `y` 的变量，类型为 `u32`（32位无符号整数）。
   * 使用 `as` 关键字进行类型转换，将 `x`（类型为 `i32`）的值转换为 `u32` 类型。
   * 将转换后的值赋给 `y`。

因此，经过这两行代码后，你会得到两个变量：`x` 的类型为 `i32`，值为 `100`；`y` 的类型为 `u32`，值也为 `100`。

需要注意的是，使用 `as` 进行类型转换可能会有风险，特别是当转换的值超出目标类型的范围时。但在这个特定的例子中，转换是安全的，因为 `100` 是 `i32` 和 `u32` 都可以表示的值

在 Rust 中，类型转换是一个非常受限制的操作，因为 Rust 高度重视类型安全。不像一些其他语言可以随意地在不同类型之间进行转换，Rust 需要明确的转换。

以下是 Rust 中一些常见的类型转换方法：

1.  **基本类型之间的转换**： 使用 `as` 关键字可以在基本数值类型之间进行转换：

    ```
    let i = 42;
    let f = i as f64;
    ```
2.  **字符串到数字**： 使用 `parse` 方法将字符串转换为数字：

    ```
    let s = "42";
    let i: i32 = s.parse().expect("Not a valid number");
    ```
3.  **数字到字符串**： 使用 `to_string` 方法将数字转换为字符串：

    ```
    let i = 42;
    let s = i.to_string();
    ```
4.  **引用和原始指针之间的转换**：

    ```
    let s = "hello";
    let p: *const str = s as *const str;
    let r: &str = unsafe { &*p };
    ```
5.  **`From` 和 `Into` trait**： Rust 提供了 `From` 和 `Into` 两个 trait 来帮助进行更复杂的类型转换。当你为一个类型实现了 `From` trait，你同时也获得了 `Into` trait 的实现。

    ```
    impl From<Foo> for Bar {
        fn from(foo: Foo) -> Self {
            // ... convert from Foo to Bar
        }
    }
    ​
    let foo = Foo { /* ... */ };
    let bar: Bar = foo.into();
    ```
6.  **`TryFrom` 和 `TryInto` trait**： 对于可能失败的转换，可以使用 `TryFrom` 和 `TryInto` trait。它们的方法返回一个 `Result`。

    ```
    rustCopy codeuse std::convert::TryFrom;
    ​
    impl TryFrom<Foo> for Bar {
        type Error = SomeError;
        
        fn try_from(foo: Foo) -> Result<Self, Self::Error> {
            // ... attempt to convert from Foo to Bar
        }
    }
    ```

在进行类型转换时，总是需要小心并确保转换是安全的。尤其是当使用 `as` 关键字时，因为它可能会进行不安全的转换，如整数溢出或裁剪。