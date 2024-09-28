# pattern matching

模式匹配是 Rust 中的一个功能强大的特性，允许你检查一个值的结构并据此执行代码。模式匹配经常与 match 表达式和 if let 结构一起使用。

以下是模式匹配的一些关键点：

**基本匹配**

最简单的模式匹配形式是 match 表达式。

```
let x = 5;
​
match x {
    1 => println!("one"),
    2 => println!("two"),
    3 => println!("three"),
    4 => println!("four"),
    5 => println!("five"),
    _ => println!("something else"),
}
```

在上述代码中，x 的值与每个模式进行匹配，当找到匹配项时，执行相应的代码。下划线 \_ 是一个通配符模式，匹配任何值。

**解构/拆分**

模式可以用来解构结构体、枚举、元组和引用。

模式匹配不仅可以用于简单的值比较，还可以用于拆解（或解构）更复杂的数据类型，从而让你可以直接访问其内部的值。这使得模式匹配成为一个非常强大的工具，尤其是在处理复杂数据结构时。

* **解构结构体**:

```
struct Point {
    x: i32,
    y: i32,
}
​
let p = Point { x: 3, y: 4 };
​
match p {
    Point { x, y } => println!("x: {}, y: {}", x, y),
}
```

在上面的代码中，`Point { x, y }` 是一个模式，它匹配任何 `Point` 结构体，并解构它，将其字段绑定到变量 `x` 和 `y`。

* **解构枚举**:

```
enum Option<T> {
    Some(T),
    None,
}
​
let some_value = Some(5);
​
match some_value {
    Some(num) => println!("Got a number: {}", num),
    None => println!("Got nothing"),
}
```

在这里，`Some(num)` 是一个模式，它匹配 `Some` 变体并解构其内部值。

* **解构元组**:

```
let tuple = (1, "hello");
​
match tuple {
    (a, b) => println!("Got: {} and {}", a, b),
}
```

这里，`(a, b)` 是一个模式，它匹配任何两元素的元组，并解构它。

**以引用方式匹配**

在 Rust 中，当你使用模式匹配对引用进行匹配时，你可能需要考虑两个方面：匹配引用本身，以及匹配引用指向的值。这时，你可以结合使用模式和解引用来达到目的。

* **匹配引用**：

直接匹配引用本身。

```
let x = &5;
​
match x {
    &v => println!("x is a reference to {}", v),
}
```

在这个例子中，我们使用 `&v` 的模式匹配引用，并将其解构为 `v`。

* **使用解引用和匹配**：

使用 `*` 进行解引用，并与模式匹配结合使用。

```
let x = &5;
​
match *x {
    v => println!("x is referencing {}", v),
}
```

这里，我们先使用 `*x` 进行解引用，然后使用模式 `v` 进行匹配。

* **在 `match` 中使用解引用**：

在 `match` 表达式中，你可以直接使用解引用的模式。

```
let x = &5;
​
match x {
    &v => println!("x is a reference to {}", v),
}
```

注意，这与第一个例子相同，但在更复杂的模式中，这种方式可能更有用。

**if let语句**

解释下："语法糖"（Syntactic sugar）是编程语言中的一个术语，指的是那些没有引入新的功能，但可以让代码更易读或更简洁的语法。语法糖的存在是为了使编程更加方便，使代码更加直观和易于理解。

换句话说，语法糖是一种便捷的编写方式，它只是现有功能的另一种表示。在编译或解释代码时，这种语法通常会被转换为更基础的、标准的语法。

`if let` 语句是 Rust 中的一个语法糖，它允许你结合 `if` 语句和模式匹配。它特别适用于当你只关心一种匹配情况，并想在这种情况下执行某个代码块时。

`if let` 的基本形式如下：

```
if let PATTERN = EXPRESSION {
    // 代码块
}
```

其中，`PATTERN` 是你想匹配的模式，`EXPRESSION` 是你想匹配的表达式。

让我们看一些实际的例子：

1.  **匹配 `Option`**：

    ```
    let some_value = Some(5);
    ​
    if let Some(x) = some_value {
        println!("Got a value: {}", x);
    }
    ```

    这里，只有当 `some_value` 是 `Some` 变体时，代码块才会执行。`x` 被绑定到 `Some` 内的值，并在代码块中使用。
2.  **与 `else` 结合使用**：

    ```
    let some_value = None;
    ​
    if let Some(x) = some_value {
        println!("Got a value: {}", x);
    } else {
        println!("Didn't match Some");
    }
    ```

    如果 `if let` 的模式不匹配，你可以使用 `else` 分支。
3.  **匹配枚举**：

    ```
    enum Message {
        Hello(String),
        Bye,
    }
    ​
    let msg = Message::Hello(String::from("World"));
    ​
    if let Message::Hello(s) = msg {
        println!("Hello, {}", s);
    }
    ```

这里，我们只在 `msg` 是 `Message::Hello` 变体时执行代码块。

`if let` 的主要优势是它提供了一个简洁的方式来处理只关心的一种匹配情况，而无需编写完整的 `match` 语句。

毕竟是语法糖，if let 和match用法是一样的，区别就在于if let更为精简

```
if let Some(x) = some_option {
    println!("Got a value: {}", x);
}
```

这个 `if let` 语句可以用 `match` 语句重写为：

```
atch some_option {
    Some(x) => println!("Got a value: {}", x),
    _ => {}
}
```

在这个 `match` 版本中，我们显式地处理了 `Some(x)` 和所有其他可能的模式（使用 `_` 通配符）。但是，`if let` 提供了一种更简洁的方式来处理我们关心的特定模式，而忽略其他所有模式。

**while let语句**

while let 是 Rust 中的另一个结合了模式匹配和循环的语法糖。它允许你在某个模式匹配成功的情况下持续执行循环体。只要模式匹配成功，循环就会继续；一旦模式匹配失败，循环就会停止。

while let 的基本结构如下：

```
while let PATTERN = EXPRESSION {
    // 代码块
}
```

其中，PATTERN 是你想匹配的模式，而 EXPRESSION 是你想匹配的表达式。

让我们看一些实际的例子：

**使用 Option：**

假设我们有一个 Vec 并使用 pop 方法。pop 返回一个 Option：如果 Vec 为空，它返回 None；否则，它返回 Some(T)，其中 T 是 Vec 的最后一个元素。

```
let mut stack = vec![1, 2, 3, 4, 5];

while let Some(top) = stack.pop() {
    println!("Popped value: {}", top);
}
```

这里，while let 循环会持续执行，直到 stack.pop() 返回 None，即 stack 为空。

**解构枚举：**

假设我们有一个表示事件的枚举，我们想从一个队列中取出并处理这些事件，直到遇到特定的事件。

```
enum Event {
    Continue(i32),
    Stop,
}
​
let mut events = vec![Event::Continue(5), Event::Continue(10), Event::Stop, Event::Continue(15)];
​
while let Some(Event::Continue(value)) = events.pop() {
    println!("Got a continue event with value: {}", value);
}
```

这里，while let 循环会处理 Event::Continue 事件，直到遇到不是 Event::Continue 的事件或 events 为空。

**内部绑定**

在 Rust 中，当我们谈论模式匹配时，有时我们会遇到需要访问内部值的情况。内部绑定允许我们在一个模式中同时匹配一个值的结构和捕获其内部值。

内部绑定的主要用途是在一个模式中匹配一个值，然后在后续的代码中使用该值。

考虑以下例子：

```
enum Message {
    Hello { id: i32 },
}
​
let msg = Message::Hello { id: 5 };
​
match msg {
    Message::Hello { id: inner_id } if inner_id > 5 => {
        println!("Hello with an id greater than 5! Got: {}", inner_id);
    },
    Message::Hello { id: _ } => {
        println!("Hello with some id!");
    },
}
```

在上述代码中，我们使用内部绑定来捕获 `id` 字段的值并将其绑定到 `inner_id` 变量。然后，我们可以在 `match` 的分支体中使用这个 `inner_id`。

这种内部绑定的能力使得我们可以在模式中进行更复杂的操作，比如在 `if` 守卫中检查捕获的值。

总的来说，内部绑定是 Rust 中模式匹配的一个强大特性，允许我们在匹配值的结构的同时，捕获并在后续的代码中使用其内部的值。

**模式匹配的穷尽性**

在 Rust 中，模式匹配的穷尽性（Exhaustiveness）是指必须处理所有可能的情况，确保没有遗漏。这是 Rust 的一个重要特性，因为它确保了代码的健壮性和安全性。

当使用 `match` 语句进行模式匹配时，Rust 编译器会检查所有可能的模式是否都被考虑到了。如果有遗漏，编译器会报错。

例如，考虑一个简单的 `enum`：

```
enum Color {
    Red,
    Green,
    Blue,
}
```

当我们使用 `match` 语句进行模式匹配时，我们必须处理所有的 `Color` 变体：

```
let my_color = Color::Green;
​
match my_color {
    Color::Red => println!("It's red!"),
    Color::Green => println!("It's green!"),
    Color::Blue => println!("It's blue!"),
}
```

如果我们遗漏了任何一个变体，例如：

```
match my_color {
    Color::Red => println!("It's red!"),
    Color::Green => println!("It's green!"),
    // Color::Blue 没有被处理
}
```

Rust 编译器会报错，因为不是所有的情况都被考虑到了。

此外，Rust 提供了一个 `_` 通配符，它可以匹配任何值。这在我们不关心某些模式时非常有用。但要小心使用，确保不要意外地忽略了重要的模式。

```
match my_color {
Color::Red => println!("It's red!"),
    _ => println!("It's some other color!"),
}
```

在这个例子中，除了 `Red` 以外的所有颜色都会匹配 `_` 模式。

**模式匹配的使用场合**

1.  **匹配枚举变体**：模式匹配经常用于处理 `enum` 类型，因为你可以轻松地区分不同的变体并处理它们。

    ```
    enum Message {
        Quit,
        Move { x: i32, y: i32 },
        Write(String),
        ChangeColor(i32, i32, i32),
    }
    ​
    match msg {
        Message::Quit => println!("The Quit variant"),
        Message::Move { x, y } => println!("Move in the x: {} y: {}", x, y),
        Message::Write(s) => println!("Text message: {}", s),
        Message::ChangeColor(r, g, b) => println!("Change color to red: {}, green: {}, blue: {}", r, g, b),
    }
    ```
2.  **解构结构体和元组**：可以使用模式匹配来解构和访问结构体或元组的值。

    ```
    let point = (3, 5);
    match point {
        (0, 0) => println!("Origin"),
        (x, 0) => println!("On the x-axis at x = {}", x),
        (0, y) => println!("On the y-axis at y = {}", y),
        (x, y) => println!("Other point at x = {}, y = {}", x, y),
    }
    ```
3.  **解构引用**：当你处理引用时，模式匹配允许你同时匹配和解引用值。

    ```
    match &some_value {
        &Some(x) => println!("Got a value: {}", x),
        &None => println!("No value"),
    }
    ```
4. **处理 `Option` 和 `Result` 类型**：这两种类型经常与模式匹配一起使用，使得处理可能的错误或缺失值变得简单明了。
5. **`if let` 和 `while let` 表达式**：这两种表达式都是基于模式匹配的，它们允许你在特定情况下简化代码。
6.  **使用守卫进行条件匹配**：你可以在模式匹配中加入额外的条件，使匹配更加灵活。

    ```
    let num = Some(4);
    match num {
        Some(x) if x < 5 => println!("less than five: {}", x),
        Some(x) => println!("{}", x),
        None => (),
    }
    ```
7. **匹配字面值**：你可以直接匹配特定的字面值，如数字、字符或字符串

**for循环模式匹配**

在 Rust 中，`for` 循环可以与模式匹配结合使用，从而允许你在迭代集合时对每个元素进行解构。这在处理复杂的数据结构时非常有用。

以下是一些 `for` 循环与模式匹配结合使用的例子：

1.  **迭代元组的数组**：

    ```
    let points = [(1, 2), (3, 4), (5, 6)];
    for (x, y) in points.iter() {
        println!("x: {}, y: {}", x, y);
    }
    ```
2.  **迭代枚举的向量**：

    ```
    enum Message {
        Move { x: i32, y: i32 },
        Write(String),
    }
    ​
    let messages = vec![
        Message::Move { x: 1, y: 2 },
        Message::Write("Hello".to_string()),
    ];
    ​
    for message in messages.iter() {
        match message {
            Message::Move { x, y } => println!("Move to x={}, y={}", x, y),
            Message::Write(text) => println!("Text message: {}", text),
        }
    }
    ```
3.  **迭代哈希映射**：

    ```
    use std::collections::HashMap;
    ​
    let mut scores = HashMap::new();
    scores.insert("Alice", 10);
    scores.insert("Bob", 15);
    ​
    for (name, score) in &scores {
        println!("{}: {}", name, score);
    }
    ```
4.  **解构复杂的结构体**：

    ```
    struct Point {
        x: i32,
        y: i32,
    }
    ​
    let points = vec![Point { x: 1, y: 2 }, Point { x: 3, y: 4 }];
    ​
    for Point { x, y } in points.iter() {
        println!("x: {}, y: {}", x, y);
    }
    ```

**函数参数的模式匹配**

在 Rust 中，函数参数也可以使用模式进行匹配。这允许我们在函数签名中直接解构复杂的数据结构，简化函数体中的代码。以下是一些使用模式匹配的函数参数的例子：

1.  **解构元组**:

    ```
    fn print_coordinates((x, y): (i32, i32)) {
        println!("x: {}, y: {}", x, y);
    }
    ​
    print_coordinates((3, 4));  // 输出: x: 3, y: 4
    ```
2.  **解构枚举**:

    ```
    enum Message {
        Move { x: i32, y: i32 },
        Write(String),
    }
    ​
    fn handle_message(msg: Message) {
        match msg {
            Message::Move { x, y } => println!("Move to x={}, y={}", x, y),
            Message::Write(text) => println!("Text message: {}", text),
        }
    }
    ```
3.  **解构结构体**:

    ```
    struct Point {
        x: i32,
        y: i32,
    }
    ​
    fn print_point(Point { x, y }: Point) {
        println!("x: {}, y: {}", x, y);
    }
    ​
    let p = Point { x: 5, y: 7 };
    print_point(p);  // 输出: x: 5, y: 7
    ```
4.  **忽略某些值**:

    ```
    fn print_first((x, _): (i32, i32)) {
        println!("x: {}", x);
    }
    ​
    print_first((8, 9));  // 输出: x: 8
    ```

使用模式匹配的函数参数可以让我们更直接地访问数据的内部结构，而无需在函数体中进行额外的解构。这使得代码更加简洁和直观。