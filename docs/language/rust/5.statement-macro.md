# statement, macro

**语句，宏**

#### 条件语句

主要使用`if`, `else if`, 和 `else` 关键字来执行。这与许多其他编程语言的结构相似。

**基本的`if`语句**

```
if condition {
    // 代码块
}
```

**`if-else`语句**

```
if condition {
    // 代码块
} else {
    // 其他代码块
}
```

使用`else if`的多条件

```
if condition1 {
    // 代码块1
} else if condition2 {
    // 代码块2
} else {
    // 其他代码块
}
```

**示例**:

假设我们有一个变量`number`，我们想根据它的值打印不同的消息。

```
let number = 10;
​
if number < 10 {
    println!("数字小于10");
} else if number == 10 {
    println!("数字等于10");
} else {
    println!("数字大于10");
}
```

在上面的示例中，输出将是 "数字等于10"。

此外，Rust中的`if`语句也可以有一个返回值，这意味着你可以将其结果直接赋值给一个变量。例如:

```
let condition = true;
let number = if condition { 5 } else { 6 };
```

在上面的代码中，如果`condition`为`true`，则`number`的值为5，否则为6。

与c++不同的地方在于，条件部分不需要用小括号引用括起来

整个条件语句是当作一个表达式来求值的，因此每一个分支都必须是相同类型的表达式。当然，如果作为普通的条件语句来使用的话，可以令类型是（）

例：

```
if x  <= 0 {
​
println!("too small!");
​
}
​
```

#### 循环语句

rust循环主要有3种；while，loop，for

* `break`和`continue`用于改变循环中的控制流

**`while`循环: 当指定的条件为真时，这个循环会一直执行。**

```
let mut number = 5;
while number > 0 {
    println!("number 的值是: {}", number);
    number -= 1;
}
```

**`loop` 循环: 这是一个无限循环，除非使用 `break` 关键字退出。**

```
loop {
    println!("这是一个无限循环");
    // 使用 break 退出循环
    if some_condition {
        break;
    }
}
```

* **使用 `break` 退出循环**

这个示例中，当`number`等于`4`时，循环会退出。

```
for number in 1..10 {
    if number == 4 {
        println!("找到了4，退出循环！");
        break;
    }
    println!("当前的数字是: {}", number);
}
```

输出:

```
code当前的数字是: 1
当前的数字是: 2
当前的数字是: 3
找到了4，退出循环！
```

**`for` 循环: 这是一个遍历范围、迭代器或集合的循环。**

a. **范围**:

```
for number in 1..4 { // 1到3（不包括4）
    println!("number 的值是: {}", number);
}
```

b. **数组和切片**:

```
let arr = [1, 2, 3, 4, 5];
for element in arr.iter() {
    println!("数组元素: {}", element);
}
```

c. **Vector**:

```
let vec = vec![1, 2, 3, 4, 5];
for element in &vec {
    println!("Vector元素: {}", element);
}
```

1. **`for` 循环与 `enumerate()`**: 当你需要遍历集合并同时获取每个元素的索引时，这很有用。

```
let arr = ["a", "b", "c"];
for (index, value) in arr.iter().enumerate() {
    println!("索引 {}: 值 {}", index, value);
}
```

1. `continue` 和 `break` 关键字:
   * `continue`: 跳过当前迭代，并继续循环的下一次迭代。
   * `break`: 完全退出循环。

```
for number in 1..6 {
    if number == 3 {
        continue; // 当 number 为 3 时，跳过并继续下一个迭代
    }
    if number == 5 {
        break; // 当 number 为 5 时，退出循环
    }
    println!("number 的值是: {}", number);
}
```

#### `print!`和`println!`

在Rust中，`print!` 和 `println!` 是两个常用的宏，用于向控制台输出文本。它们的行为与许多其他编程语言中的 `print` 和 `println` 或 `printf` 功能类似，但有一些特定的差异。

1. **`print!` 宏**:
   * `print!` 将文本输出到控制台，但不在其后添加新行。
   * 语法: `print!("格式字符串", 参数1, 参数2, ...)`
2. **`println!` 宏**:
   * `println!` 也将文本输出到控制台，但它会在文本后面添加一个新行，所以随后的输出会从新的一行开始。
   * 语法: `println!("格式字符串", 参数1, 参数2, ...)`

**示例**:

```
rustCopy codefn main() {
    print!("这是");
    print!("一个");
    print!("例子");
    
    println!(); // 添加一个新行
​
    println!("这是");
    println!("另一个");
    println!("例子");
}
```

输出:

```
这是一个例子
这是
另一个
例子
```

与其他语言的格式化输出功能类似，你可以在字符串中使用占位符，然后在宏调用中传递要插入这些占位符位置的值。例如:

```
fn main() {
    let name = "Alice";
    let age = 30;
    println!("我的名字是{}，我{}岁了。", name, age);
}
```

输出:

```
我的名字是Alice，我30岁了。
```

请注意，由于这些都是宏而不是函数，所以在宏名称后面有一个感叹号 (`!`)。

#### format！

使用与`print！`/`println！`相同的用法来创建`String`字符串。

format! 宏在Rust中用于创建格式化的字符串，但与print!和println!不同，它不会将结果输出到控制台。相反，它返回一个表示格式化字符串的String。这允许您在其他地方使用或存储格式化的字符串。

语法:

```
let formatted_string = format!("格式字符串", 参数1, 参数2, ...);
```

示例:

```
let name = "Alice";
let age = 30;
let formatted = format!("我的名字是{}，我{}岁了。", name, age);
​
// `formatted` 现在包含 "我的名字是Alice，我30岁了。"
println!("{}", formatted);
```

使用位置参数:

```
let formatted = format!("{0} 和 {1} 喜欢 {2}", "Alice", "Bob", "巧克力");
// 输出: "Alice 和 Bob 喜欢 巧克力"
println!("{}", formatted);
```

使用命名参数:

```
let formatted = format!("{name} 是一个 {job}", name="Alice", job="工程师");
// 输出: "Alice 是一个 工程师"
println!("{}", formatted);
```

格式选项:

```
let pi = 3.141592;
let formatted = format!("{:.2}", pi); // 保留两位小数
// 输出: "3.14"
println!("{}", formatted);
```

format! 宏非常灵活，允许各种格式选项和参数类型。这使得它在需要动态构建字符串时非常有用。

\