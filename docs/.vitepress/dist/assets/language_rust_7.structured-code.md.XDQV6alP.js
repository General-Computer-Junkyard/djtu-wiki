import{_ as n,c as a,a0 as p,o as e}from"./chunks/framework.DOEc3Y8C.js";const g=JSON.parse('{"title":"structured code","description":"","frontmatter":{},"headers":[],"relativePath":"language/rust/7.structured-code.md","filePath":"language/rust/7.structured-code.md"}'),l={name:"language/rust/7.structured-code.md"};function t(i,s,o,c,d,r){return e(),a("div",null,s[0]||(s[0]=[p(`<h1 id="structured-code" tabindex="-1">structured code <a class="header-anchor" href="#structured-code" aria-label="Permalink to &quot;structured code&quot;">​</a></h1><h4 id="结构化代码" tabindex="-1">结构化代码 <a class="header-anchor" href="#结构化代码" aria-label="Permalink to &quot;结构化代码&quot;">​</a></h4><p>rust有2种创建结构化数据类形的方式：</p><ul><li>结构体<code>struct</code>：像c/c++那样的结构体，用于保存数据</li><li>枚举<code>enum</code>：像OCaml，数据可以是几种类形之一</li></ul><p>结构体和枚举都可以有若干实现块impl，用于定义相应类形的方法</p><p>结构体（Structs）和枚举（Enums）的一个特性，即它们都可以拥有多个<code>impl</code>块来定义方法。</p><ol><li><p><strong>结构体 (<code>Structs</code>)</strong>: 在Rust中，结构体是一种组合多个数据片段的方式。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct Point {</span></span>
<span class="line"><span>    x: i32,</span></span>
<span class="line"><span>    y: i32,</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>枚举 (<code>Enums</code>)</strong>: 枚举允许你定义一种数据，该数据可以取多个不同的形式。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>enum Message {</span></span>
<span class="line"><span>    Quit,</span></span>
<span class="line"><span>    Move { x: i32, y: i32 },</span></span>
<span class="line"><span>    Write(String),</span></span>
<span class="line"><span>    ChangeColor(i32, i32, i32),</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>实现块 (<code>impl</code> blocks)</strong>: <code>impl</code>块用于定义结构体或枚举的方法。你可以为一个结构体或枚举定义多个<code>impl</code>块。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>impl Point {</span></span>
<span class="line"><span>    // 这是第一个impl块</span></span>
<span class="line"><span>    fn distance_from_origin(&amp;self) -&gt; f64 {</span></span>
<span class="line"><span>        (self.x.pow(2) + self.y.pow(2)).sqrt() as f64</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>impl Point {</span></span>
<span class="line"><span>    // 这是第二个impl块</span></span>
<span class="line"><span>    fn translate(&amp;mut self, dx: i32, dy: i32) {</span></span>
<span class="line"><span>        self.x += dx;</span></span>
<span class="line"><span>        self.y += dy;</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上述代码中，我们为<code>Point</code>结构体定义了两个不同的<code>impl</code>块，每个块中有一个方法。</p></li></ol><p><strong>结构体的声明</strong></p><p><strong>具名字段的结构体 (Named Fields)</strong></p><p>这是最常见的结构体形式，其中每个字段都有一个明确的名称。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct User {</span></span>
<span class="line"><span>    username: String,</span></span>
<span class="line"><span>    email: String,</span></span>
<span class="line"><span>    age: u32,</span></span>
<span class="line"><span>    active: bool,</span></span>
<span class="line"><span>}</span></span></code></pre></div><p><strong>元组结构体 (Tuple Structs)</strong></p><p>元组结构体（Tuple Structs）是Rust中的一种特殊类型的结构体。它们的特点是字段没有具体的名称，只有类型，这种结构体看起来像元组，但它们有一个具体的类型名称。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct Color(u8, u8, u8);</span></span>
<span class="line"><span>struct Point(f64, f64);</span></span></code></pre></div><p><strong>定义元组结构体</strong> 要定义元组结构体，你需要提供一个结构体名称，后面跟随圆括号，并在圆括号内列出字段的类型。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct Point3D(f64, f64, f64);</span></span>
<span class="line"><span>struct Color(u8, u8, u8);</span></span>
<span class="line"><span>struct Pair(i32, i32);</span></span></code></pre></div><p><strong>创建元组结构体的实例</strong> 就像普通元组一样，你可以使用圆括号来创建元组结构体的实例。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let origin = Point3D(0.0, 0.0, 0.0);</span></span>
<span class="line"><span>let red = Color(255, 0, 0);</span></span>
<span class="line"><span>let coordinates = Pair(10, 20);</span></span></code></pre></div><p><strong>访问元组结构体的字段</strong> 你可以使用.后跟字段的索引来访问元组结构体的字段，就像访问普通元组的元素一样。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let x = origin.0;</span></span>
<span class="line"><span>let y = origin.1;</span></span>
<span class="line"><span>let z = origin.2;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let r = red.0;</span></span>
<span class="line"><span>let g = red.1;</span></span>
<span class="line"><span>let b = red.2;</span></span></code></pre></div><p><strong>为什么使用元组结构体？</strong> 尽管元组结构体和普通元组在许多方面都很相似，但有时你可能希望为某个特定的数据结构提供一个明确的名称，而不仅仅是一个匿名元组。元组结构体允许你为数据提供类型别名，这有助于提高代码的可读性和意图的清晰性。</p><p>例如，虽然(f64, f64, f64)和(u8, u8, u8)在类型上是不同的，但它们在语义上没有明确的区别。使用Point3D和Color作为元组结构体名称，可以明确地区分这两种数据结构的用途和意图。</p><p><strong>单元结构体 (Unit Structs)</strong></p><p>单元结构体（Unit Structs）是Rust中的一种特殊类型的结构体。与元组结构体和常规结构体不同，单元结构体没有任何字段。它们的定义不包含任何内容，只是一个名称。</p><p><strong>定义单元结构体</strong></p><p>单元结构体的定义非常简单，只需要一个名称：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct MyUnitStruct;</span></span></code></pre></div><p>如你所见，它看起来就像是一个结构体名称，后面跟着一个分号，没有任何其他内容。</p><p>为什么使用单元结构体？</p><p>可能会问，没有字段的结构体有什么用呢？以下是一些使用单元结构体的常见场景：</p><ol><li><p><strong>类型标记</strong>：单元结构体可以作为一个明确的类型标记。这在泛型编程中尤其有用，当你需要一个没有数据的类型，但仍然想区分其他类型时。</p></li><li><p><strong>Trait实现</strong>：在Rust中，你可以为任何类型实现trait，包括单元结构体。这意味着你可以使用单元结构体来组织特定的方法或行为，而不必关心数据。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>trait SayHello {</span></span>
<span class="line"><span>    fn say_hello(&amp;self);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>struct HelloUnit;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>impl SayHello for HelloUnit {</span></span>
<span class="line"><span>    fn say_hello(&amp;self) {</span></span>
<span class="line"><span>        println!(&quot;Hello from HelloUnit!&quot;);</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span></code></pre></div></li><li><p><strong>Opaque类型</strong>：在某些情况下，你可能想隐藏实际的数据实现，而只提供一个类型标识。单元结构体可以在这样的场景中被用作一个不透明的类型标识。</p></li></ol><p><strong>创建和使用单元结构体</strong></p><p>创建单元结构体的实例非常简单：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let unit_instance = MyUnitStruct;</span></span></code></pre></div><p>由于单元结构体没有任何字段，所以你不能访问或修改其内容。然而，如果你为它实现了方法或trait，你可以调用这些方法。</p><p>总之，虽然单元结构体可能看起来不像传统的数据结构，但它们在Rust中是有用的，尤其是在需要类型标记或特定的trait实现，但不需要实际数据时。</p><p><strong>创建结构体实例</strong></p><p>一旦你声明了结构体，你可以使用它来创建该结构体的实例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>// 使用具名字段的结构体</span></span>
<span class="line"><span>let user1 = User {</span></span>
<span class="line"><span>    username: String::from(&quot;Alice&quot;),</span></span>
<span class="line"><span>    email: String::from(&quot;alice@example.com&quot;),</span></span>
<span class="line"><span>    age: 30,</span></span>
<span class="line"><span>    active: true,</span></span>
<span class="line"><span>};</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 使用元组结构体</span></span>
<span class="line"><span>let black = Color(0, 0, 0);</span></span>
<span class="line"><span>let origin = Point(0.0, 0.0);</span></span></code></pre></div><p><strong>可变性</strong></p><p>默认情况下，结构体实例是不可变的。但是，通过添加<code>mut</code>关键字，你可以创建一个可变的结构体实例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let mut user2 = User {</span></span>
<span class="line"><span>    username: String::from(&quot;Bob&quot;),</span></span>
<span class="line"><span>    email: String::from(&quot;bob@example.com&quot;),</span></span>
<span class="line"><span>    age: 25,</span></span>
<span class="line"><span>    active: false,</span></span>
<span class="line"><span>};</span></span>
<span class="line"><span>user2.age = 26;  // 修改age字段的值</span></span></code></pre></div><p>这就是Rust中结构体的基本声明和使用方法。结构体是Rust中非常强大的自定义数据类型，允许你将相关数据组合在一起，并为其定义方法和关联函数</p><p><strong>&lt;!--在Rust中，当我们谈论&quot;可变性&quot;时，我们实际上是指变量的绑定，而不是结构体本身。--&gt;</strong></p><p>结构体定义（无论是常规结构体、元组结构体还是单元结构体）本身并不具有可变性。它只是定义了一个数据结构。但当你创建一个结构体的实例并为其绑定一个变量时，这个绑定可以是可变的或不可变的。</p><p>以下是如何创建可变和不可变的结构体实例的示例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct Point {</span></span>
<span class="line"><span>    x: i32,</span></span>
<span class="line"><span>    y: i32,</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 创建一个不可变的Point实例</span></span>
<span class="line"><span>let p1 = Point { x: 0, y: 0 };</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 创建一个可变的Point实例</span></span>
<span class="line"><span>let mut p2 = Point { x: 1, y: 1 };</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 由于p2是可变的，我们可以修改它的字段</span></span>
<span class="line"><span>p2.x = 2;</span></span>
<span class="line"><span>p2.y = 2;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>// 但我们不能修改p1的字段，因为它是不可变的</span></span>
<span class="line"><span>// p1.x = 3;  // 这会导致编译错误</span></span></code></pre></div><p>所以，结论是：你可以为结构体实例的绑定添加mut关键字以使其可变，这允许你修改该实例的字段。但这并不改变结构体定义本身的性质。</p><p><strong>结构体的访问</strong></p><p>在Rust中，结构体的字段默认是私有的，这意味着它们只能在定义结构体的当前模块中访问。</p><p>但你可以使用<code>pub</code>关键字来明确地使字段公开，从而允许外部模块访问它。</p><ol><li><strong>默认的私有字段</strong></li></ol><p>当你创建一个结构体时，它的字段默认是私有的。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>mod my_module {</span></span>
<span class="line"><span>    pub struct Person {</span></span>
<span class="line"><span>        name: String,</span></span>
<span class="line"><span>        age: u32,</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>    pub fn create_person(name: String, age: u32) -&gt; Person {</span></span>
<span class="line"><span>        Person { name, age }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上面的代码中，<code>Person</code>结构体的<code>name</code>和<code>age</code>字段是私有的。这意味着你不能从<code>my_module</code>模块外部直接访问这些字段。</p><ol start="2"><li><strong>使用<code>pub</code>使字段公开</strong></li></ol><p>通过在字段名前加上<code>pub</code>关键字，你可以使字段公开。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>mod my_module {</span></span>
<span class="line"><span>pub struct Person {</span></span>
<span class="line"><span>        pub name: String,</span></span>
<span class="line"><span>        pub age: u32,</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>    pub fn create_person(name: String, age: u32) -&gt; Person {</span></span>
<span class="line"><span>        Person { name, age }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>现在，<code>name</code>和<code>age</code>字段是公开的，可以从<code>my_module</code>模块外部访问。</p><ol start="3"><li><strong>访问结构体的字段</strong></li></ol><p>访问结构体的字段非常简单。只需使用<code>.</code>符号，后跟字段的名称。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let person = my_module::create_person(String::from(&quot;Alice&quot;), 30);</span></span>
<span class="line"><span>println!(&quot;Name: {}&quot;, person.name);</span></span>
<span class="line"><span>println!(&quot;Age: {}&quot;, person.age);</span></span></code></pre></div><p><strong>注意：</strong></p><ul><li>尽管可以通过<code>pub</code>关键字公开结构体的字段，但在许多情况下，为了封装和保护数据，最好保持字段为私有，并提供公开的方法来获取或修改这些字段。</li><li>当你在定义结构体的模块中工作时，可以直接访问其私有字段。</li></ul><p><strong>结构体的访问权限</strong></p><p>在 Rust 中，访问权限（或称为可见性）决定了结构体、其字段、函数、模块等元素是否可以从其他模块或外部代码中访问。以下是关于结构体和其访问权限的一些关键点：</p><ol><li><p><strong>默认私有</strong>：Rust 中的所有项（包括结构体字段）默认都是私有的。这意味着，除非你明确地使用 <code>pub</code> 关键字来公开它们，否则它们只能在定义它们的模块中访问。</p></li><li><p><strong>结构体的公开性</strong>：如果你想从外部模块或代码中创建或明确引用结构体，你需要使用 <code>pub</code> 关键字公开它：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>pub struct Person {</span></span>
<span class="line"><span>    name: String,</span></span>
<span class="line"><span>    age: u32,</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在这里，<code>Person</code> 结构体是公开的，但其字段 <code>name</code> 和 <code>age</code> 仍然是私有的。</p></li><li><p><strong>字段的公开性</strong>：如果你想从外部访问或修改结构体的某个字段，你需要公开那个字段：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>pub struct Person {</span></span>
<span class="line"><span>    pub name: String,</span></span>
<span class="line"><span>    pub age: u32,</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>现在，<code>name</code> 和 <code>age</code> 字段都是公开的，可以从任何地方访问和修改。</p></li><li><p><strong>在模块中的访问权限</strong>：你可以在模块中定义结构体，并根据需要设置其可见性。例如，如果你在一个子模块中定义一个私有结构体，它只能在该子模块中访问，而不能在父模块或其他模块中访问。</p></li><li><p><strong>关联函数和方法的公开性</strong>：与结构体和其字段类似，结构体的关联函数和方法也是默认私有的。如果你想从外部调用它们，你需要使用 <code>pub</code> 关键字公开它们。</p></li></ol><p><strong>结构体的更新语法</strong></p><p>结构体的更新语法允许从一个已存在的结构体实例创建一个新实例，同时只改变部分字段的值。这种语法非常有用，特别是当结构体有很多字段，而你只想修改其中的一小部分时。</p><p>使用 <code>..</code> 语法可以指定你希望使用另一个结构体实例的字段值。</p><p>以下是一个例子来说明这种语法：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>struct Point {</span></span>
<span class="line"><span>    x: i32,</span></span>
<span class="line"><span>    y: i32,</span></span>
<span class="line"><span>    z: i32,</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>fn main() {</span></span>
<span class="line"><span>    let p1 = Point { x: 1, y: 2, z: 3 };</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>    // 使用结构体更新语法从p1创建p2，只改变x字段</span></span>
<span class="line"><span>    let p2 = Point { x: 5, ..p1 };</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>    println!(&quot;p2: x = {}, y = {}, z = {}&quot;, p2.x, p2.y, p2.z);</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在上面的例子中，<code>p2</code> 会有 <code>x</code> 值为 5，而 <code>y</code> 和 <code>z</code> 的值会与 <code>p1</code> 相同。</p>`,73)]))}const h=n(l,[["render",t]]);export{g as __pageData,h as default};
