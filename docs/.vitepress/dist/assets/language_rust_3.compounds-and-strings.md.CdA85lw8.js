import{_ as n,c as a,a0 as p,o as e}from"./chunks/framework.DOEc3Y8C.js";const g=JSON.parse('{"title":"Compounds and strings","description":"","frontmatter":{},"headers":[],"relativePath":"language/rust/3.compounds-and-strings.md","filePath":"language/rust/3.compounds-and-strings.md"}'),l={name:"language/rust/3.compounds-and-strings.md"};function t(i,s,o,c,r,d){return e(),a("div",null,s[0]||(s[0]=[p(`<h1 id="compounds-and-strings" tabindex="-1">Compounds and strings <a class="header-anchor" href="#compounds-and-strings" aria-label="Permalink to &quot;Compounds and strings&quot;">​</a></h1><p><strong>复合和字符串</strong></p><h4 id="复合" tabindex="-1">复合 <a class="header-anchor" href="#复合" aria-label="Permalink to &quot;复合&quot;">​</a></h4><p>符合类型是由其他类型组成，经典例子就是结构体<code>struct</code>和枚举<code>enum</code>。Rust 主要有两种复合类型：元组（tuples）和数组（arrays）。</p><p><strong>元组 (Tuples)</strong></p><ul><li>元组是一个可以包含多个类型的值的集合。</li><li>元组的大小在定义时是固定的。</li><li>访问元组中的值是通过索引来实现的。</li></ul><p>例如：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let tup: (i32, f64, u8) = (500, 6.4, 1);</span></span></code></pre></div><p>&lt;!--在上面的代码中，\`tup\` 是一个包含三个不同类型的元素的元组。--&gt;</p><p>（）它是一个特殊的类型，表示没有任何有意义的值。这可以看作是一个零元素的元组。</p><p><strong>数组 (Arrays)</strong></p><ul><li>数组是一个包含多个相同类型值的集合。</li><li>数组的大小在定义时也是固定的。</li><li>数组用于需要将多个相同类型的值存储在一起的情况，而且你知道在编译时它们的确切数量。</li></ul><p>例如：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let arr: [i32; 5] = [1, 2, 3, 4, 5];</span></span></code></pre></div><p>&lt;!--&lt;u&gt;在上面的代码中，\`arr\` 是一个包含五个 \`i32\` 类型元素的数组&lt;/u&gt;--&gt;</p><p>数组和元组之间的联系和区别</p><p>区别： 元素类型：</p><p>数组：数组中的所有元素都必须是相同的类型。 元组：元组中的元素可以是不同的类型。 大小：</p><p>数组：数组的大小是固定的，这意味着一旦定义了数组的长度，你不能改变它。 元组：元组的大小也是固定的，一旦定义，你不能更改它。但与数组不同的是，元组的长度不是其定义的一部分，所以你可以有多个不同长度的元组而不需要使用不同的类型名称。 访问元素：</p><p>数组：你使用索引来访问数组的元素，例如 arr[0]。 元组：你可以使用点号和索引来访问元组的元素，例如 tup.0。 主要用途：</p><p>数组：当你需要存储多个相同类型的元素时使用数组。 元组：当你需要组合不同类型的值时使用元组。 联系： 固定大小：两者都有固定的大小，这意味着一旦定义，你不能更改它们的大小。</p><p>在栈上存储：数组和元组中的元素通常都是在栈上分配的，除非它们被包含在堆上分配的数据结构中。</p><p>模式匹配：在 Rust 中，你可以使用模式匹配来解构数组和元组，这使得你可以方便地访问它们的内容。</p><p>内存连续性：在内存中，数组和元组的元素都是连续存储的。</p><p><strong>复合的作用</strong>:</p><ul><li><strong>组织数据</strong>：复合类型允许你将相关的数据组织在一起，而不是每个数据都单独声明为变量。</li><li><strong>固定大小的集合</strong>：与其他语言中的列表或动态数组不同，Rust 的元组和数组在定义时大小是固定的，这意味着它们在堆上不会动态增长或缩小。</li><li><strong>类型安全</strong>：特别是在元组中，你可以在一个集合中存储不同类型的值，并在编译时确保每个元素的类型正确。</li><li><strong>效率</strong>：由于它们的大小是固定的，数组特别是在性能关键的场景中，可以提供更高的效率，因为它们可以在栈上分配。</li></ul><p>Rust 中的复合类型提供了一种在一个集合中组织和处理多个值的方式，这些值可以是相同的类型（如数组）或不同的类型（如元组）</p><p><strong>rust语言圣经中举例：</strong></p><p>例如平面上的一个点 <code>point(x, y)</code>，它由两个数值类型的值 <code>x</code> 和 <code>y</code> 组合而来。我们无法单独去维护这两个数值，因为单独一个 <code>x</code> 或者 <code>y</code> 是含义不完整的，无法标识平面上的一个点，应该把它们看作一个整体去理解和处理。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>#![allow(unused_variables)]</span></span>
<span class="line"><span>type File = String;</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>fn open(f: &amp;mut File) -&gt; bool {</span></span>
<span class="line"><span>    true</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>fn close(f: &amp;mut File) -&gt; bool {</span></span>
<span class="line"><span>    true</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>#[allow(dead_code)]</span></span>
<span class="line"><span>fn read(f: &amp;mut File, save_to: &amp;mut Vec&lt;u8&gt;) -&gt; ! {</span></span>
<span class="line"><span>    unimplemented!()</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>fn main() {</span></span>
<span class="line"><span>    let mut f1 = File::from(&quot;f1.txt&quot;);</span></span>
<span class="line"><span>    open(&amp;mut f1);</span></span>
<span class="line"><span>    //read(&amp;mut f1, &amp;mut vec![]);</span></span>
<span class="line"><span>    close(&amp;mut f1);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span></code></pre></div><p>代码解释：</p><p>&lt;!--//#![allow(unused_variables)] 属性标记，该标记会告诉编译器忽略未使用的变量，不要抛出 warning 警告，read 函数也非常有趣，它返回一个 ! 类型，这个表明该函数是一个发散函数，不会返回任何值，包括 ()。unimplemented!() 告诉编译器该函数尚未实现，unimplemented!() 标记通常意味着我们期望快速完成主要代码，回头再通过搜索这些标记来完成次要代码，类似的标记还有 todo!()，当代码执行到这种未实现的地方时，程序会直接报错。你可以反注释 read(&amp;mut f1, &amp;mut vec![])。--&gt;</p><h4 id="字符串" tabindex="-1">字符串 <a class="header-anchor" href="#字符串" aria-label="Permalink to &quot;字符串&quot;">​</a></h4><p>Rust 中的字符串处理有些独特，因为它为了确保内存安全和并发安全采取了严格的措施。在 Rust 中，主要有两种字符串类型：<code>String</code> 和 <code>str</code>。这两种类型分别对应于可变字符串和不可变字符串。</p><p><strong>切片slice</strong></p><p>在开始记笔记之前先引用”rust语言圣经“中一个案例去了解下切片。</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>首先来看段很简单的代码：（greet 函数接受一个字符串类型的 name 参数，然后打印到终端控制台中，非常好理解）</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>fn main() {</span></span>
<span class="line"><span>  let my_name = &quot;Pascal&quot;;</span></span>
<span class="line"><span>  greet(my_name);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>fn greet(name: String) {</span></span>
<span class="line"><span>  println!(&quot;Hello, {}!&quot;, name);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>你们猜猜，这段代码能否通过编译？</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>error[E0308]: mismatched types</span></span>
<span class="line"><span> --&gt; src/main.rs:3:11</span></span>
<span class="line"><span>  |</span></span>
<span class="line"><span>3 |     greet(my_name);</span></span>
<span class="line"><span>  |           ^^^^^^^</span></span>
<span class="line"><span>  |           |</span></span>
<span class="line"><span>  |           expected struct \`std::string::String\`, found \`&amp;str\`</span></span>
<span class="line"><span>  |           help: try using a conversion method: \`my_name.to_string()\`</span></span>
<span class="line"><span>  </span></span>
<span class="line"><span>error: aborting due to previous error</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>Bingo，果然报错了，编译器提示 greet 函数需要一个 String 类型的字符串，却传入了一个 &amp;str 类型的字符串，相信读者心中现在一定有几头草泥马呼啸而过，怎么字符串也能整出这么多花活？</span></span>
<span class="line"><span>在讲解字符串之前，先来看看什么是切片?</span></span></code></pre></div><p>切片的半官方定义是：对集合中一系列连续元素的引用。</p><p>是一个没有所有权的数据类型，对于字符串而言，切片就是对 String 类型中某一部分的引用。</p><p>再者说：切片可以用于访问数组、字符串或其他集合的一部分数据，而无需拷贝。它们常用于函数参数，以避免数据的所有权移动或不必要的数据复制。</p><p>就是大致这样：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>#![allow(unused)]</span></span>
<span class="line"><span>fn main() {</span></span>
<span class="line"><span>let s = String::from(&quot;hello world&quot;);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let hello = &amp;s[0..5];</span></span>
<span class="line"><span>let world = &amp;s[6..11];</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>代码解释</p><p>&lt;!--#![allow(unused)] 是一个编译器指令，告诉 Rust 的编译器忽略未使用的代码的警告。在这种情况下，由于你只是声明了 hello 和 world 但没有实际使用它们，所以编译器会产生警告。通过添加 #![allow(unused)]，这些警告将被抑制。这个属性通常在开发过程中的某些阶段使用，例如当你正在编写一部分代码但尚未完成，或者当你想临时禁用某些警告时。在生产代码中，通常建议处理这些警告，而不是简单地禁用它们，以确保代码的质量和维护性。--&gt;</p><p>&lt;!--这段代码写的是创建了一个 String 类型的变量 s，并初始化为 &quot;hello world&quot;。--&gt; &lt;!--使用字符串切片语法，从 s 中提取 &quot;hello&quot; 和 &quot;world&quot;，并分别将它们存储在变量 hello 和 world 中。--&gt;</p><p>&lt;!--创建切片语法就在于此，而使用方括号包括了什么时候和什么时候结束，即[ 开始索引..结束索引 ]。这里的开始索引就是在说明第一个元素开始的位置，结束索引便是你想结束的位置--&gt;</p><p><img src="https://pic1.zhimg.com/80/v2-69da917741b2c610732d8526a9cc86f5_1440w.jpg" alt="img"></p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>rust语言圣经解释：</span></span>
<span class="line"><span>hello 没有引用整个 String s，而是引用了 s 的一部分内容，通过 [0..5] 的方式来指定。</span></span>
<span class="line"><span>对于 let world = &amp;s[6..11]; 来说，world 是一个切片，该切片的指针指向 s 的第 7 个字节(索引从 0 开始, 6 是第 7 个字节)，且该切片的长度是 5 个字节。</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>rust程序设计与语言解释： </span></span>
<span class="line"><span>这类似于引用整个 String 不过带有额外的 [0..5] 部分。它不是对整个 String 的引用，而是对部分 String 的引用。</span></span>
<span class="line"><span>可以使用一个由中括号中的 [starting_index..ending_index] 指定的 range 创建一个 slice，其中 starting_index 是 slice 的第一个位置，ending_index 则是 slice 最后一个位置的后一个值。在其内部，slice 的数据结构存储了 slice 的开始位置和长度，长度对应于 ending_index 减去 starting_index 的值。所以对于 let world = &amp;s[6..11]; 的情况，world 将是一个包含指向 s 索引 6 的指针和长度值 5 的 slice。</span></span>
<span class="line"><span>​</span></span></code></pre></div><p>切片要只是包括<code>string</code>的最后一个字节，便可以：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let len = s.len();</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>let slice = &amp;s[4..len];</span></span>
<span class="line"><span>let slice = &amp;s[4..];</span></span></code></pre></div><p>代码解释</p><p>let s = String::from(&quot;hello&quot;);这行代码创建了一个 String 类型的变量 s 并初始化为 &quot;hello&quot;。</p><p>let len = s.len();这行代码获取字符串 s 的长度并存储在变量 len 中。对于字符串 &quot;hello&quot;，其长度为 5。</p><p>Len(): 这是一个方法调用。在这里，你调用了 String 类型的 len 方法。这个方法返回字符串中字节的数量。 对于字符串 &quot;hello&quot;，它是由5个字符组成的，而且每个字符都是一个字节（在这种情况下，因为这些字符都是 ASCII 字符），所以 s.len() 返回5。 let len = ...;:</p><p>使用 let 关键字，你创建了一个新的不可变变量 len。 你将 s.len() 的返回值（也就是5）赋给了这个新变量 len。 总的来说，第二行代码做的事情是：调用字符串 s 的 len 方法来获取其长度，并将这个长度值存储在一个新的变量 len 中。现在，变量 len 包含数字5。</p><p>let slice = &amp;s[4..len];这行代码从 s 中提取一个切片，开始于索引 4（包含）并结束于索引 len。由于 len 是 5，这意味着切片结束于字符串的末尾。结果是，slice 包含字符串 &quot;o&quot;。</p><p>let slice = &amp;s[4..];这行代码也是从 s 中提取一个切片，但结束索引被省略了，这意味着它默认切片到字符串的末尾。这与前面的切片一样，slice 包含字符串 &quot;o&quot;。</p><p>使用范围的结束索引（如 ..len）或省略结束索引（如 4..）都是达到相同效果的有效方式。省略结束索引通常更简洁，尤其是当你想切片到字符串的末尾时。</p><p><strong>切片类型&amp;String和&amp;str</strong></p><p>Rust 中有几种不同的切片类型，但最常见的是字符串切片 (&amp;str) 和数组切片 (&amp;[T])。</p><p>切片的几个关键点：</p><ul><li><strong>不拥有数据</strong>：切片只是对原始数据的引用，不拥有这些数据。这意味着切片的生命周期不能超过它引用的数据。</li><li><strong>动态大小</strong>：切片的大小在运行时是可知的，但在编译时不是。这与数组不同，数组的大小在编译时是已知的。</li><li><strong>边界检查</strong>：当使用切片时，Rust 会执行边界检查。尝试创建超出原始数据边界的切片会导致编译时错误或运行时 panic。</li></ul><p>字符串切片的类型标识是 <code>&amp;str</code>，因此我们可以这样声明一个函数，输入 <code>String</code> 类型，返回它的切片:</p><p><code>fn first_word(s: &amp;String) -&gt; &amp;str。</code></p><p>这个函数在字符串上使用切片来实现</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>​</span></span>
<span class="line"><span>fn first_word(s: &amp;String) -&gt; &amp;str {</span></span>
<span class="line"><span>    let bytes = s.as_bytes(); // 将字符串转换为字节数组</span></span>
<span class="line"><span>    for (i, &amp;byte) in bytes.iter().enumerate() {</span></span>
<span class="line"><span>        if byte == b&#39; &#39; {</span></span>
<span class="line"><span>            return &amp;s[0..i]; // 返回从开头到第一个空格之前的切片</span></span>
<span class="line"><span>        }</span></span>
<span class="line"><span>    }</span></span>
<span class="line"><span>    &amp;s[..] // 如果没有空格，则返回整个字符串的切片</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>​</span></span></code></pre></div><p>同理有了切片就可以写出这样的代码：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>fn main() {</span></span>
<span class="line"><span>    let mut s = String::from(&quot;hello world&quot;);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>    let word = first_word(&amp;s);</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>    s.clear(); // error!</span></span>
<span class="line"><span>​</span></span>
<span class="line"><span>    println!(&quot;the first word is: {}&quot;, word);</span></span>
<span class="line"><span>}</span></span>
<span class="line"><span>fn first_word(s: &amp;String) -&gt; &amp;str {</span></span>
<span class="line"><span>    &amp;s[..1]</span></span>
<span class="line"><span>}</span></span></code></pre></div><p>在报错界面会出现这样的问题</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>error[E0502]: cannot borrow \`s\` as mutable because it is also borrowed as immutable</span></span>
<span class="line"><span>  --&gt; src/main.rs:18:5</span></span>
<span class="line"><span>   |</span></span>
<span class="line"><span>16 |     let word = first_word(&amp;s);</span></span>
<span class="line"><span>   |                           -- immutable borrow occurs here</span></span>
<span class="line"><span>17 |</span></span>
<span class="line"><span>18 |     s.clear(); // error!</span></span>
<span class="line"><span>   |     ^^^^^^^^^ mutable borrow occurs here</span></span>
<span class="line"><span>19 |</span></span>
<span class="line"><span>20 |     println!(&quot;the first word is: {}&quot;, word);</span></span>
<span class="line"><span>   |                                       ---- immutable borrow later used here</span></span>
<span class="line"><span>​</span></span></code></pre></div><p>&lt;!--这里如果试图在调用 \`first_word\` 函数后使用 \`s.clear()\` 来清空字符串 \`s\`。然而，这会导致编译错误，因为在 \`first_word\` 函数中返回了字符串 \`s\` 的切片，这个切片引用了原始字符串的数据，一旦清空了原始字符串，这个切片就会变得无效。--&gt;</p><p>&lt;!--在Rust中，引用和借用规则非常严格，以确保安全性。在这种情况下，可以考虑使用不可变引用来避免这个问题，同时也需要调整 \`first_word\` 函数，以便返回一个字符串切片，而不是 \`&amp;s[..1]\` 这样的硬编码切片。--&gt;</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>rust语言圣经：</span></span>
<span class="line"><span>回忆一下借用的规则：当我们已经有了可变借用时，就无法再拥有不可变的借用。因为 clear 需要清空改变 String，因此它需要一个可变借用（利用 VSCode 可以看到该方法的声明是 pub fn clear(&amp;mut self) ，参数是对自身的可变借用 ）；而之后的 println! 又使用了不可变借用，也就是在 s.clear() 处可变借用与不可变借用试图同时生效，因此编译无法通过。</span></span>
<span class="line"><span>从上述代码可以看出，Rust 不仅让我们的 API 更加容易使用，而且也在编译期就消除了大量错误！</span></span></code></pre></div><p>之前提到过字符串字面量,但是没有提到它的类型：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let s = &quot;Hello, world!&quot;;</span></span></code></pre></div><p>实际上，<code>s</code> 的类型是 <code>&amp;str</code>，因此你也可以这样声明：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let s: &amp;str = &quot;Hello, world!&quot;;</span></span></code></pre></div><p><strong><code>&amp;String</code></strong>:</p><ul><li><code>&amp;String</code> 是对 <code>String</code> 类型的引用。<code>String</code> 是一个可变、拥有（owned）的字符串类型，而 <code>&amp;String</code> 表示对这个可变字符串的不可变引用。</li><li>因为 <code>&amp;String</code> 是对 <code>String</code> 的引用，所以它可以访问 <code>String</code> 上的所有方法和功能。您可以使用 <code>&amp;String</code> 来读取字符串数据，但不能修改它，因为它是不可变的引用。</li></ul><p><strong><code>&amp;str</code></strong>:</p><ul><li><code>&amp;str</code> 是对字符串切片的引用。字符串切片是一个不可变的引用，可以引用一部分字符串数据，而不需要拥有整个字符串。</li><li><code>&amp;str</code> 可以引用任何包含文本数据的类型，包括 <code>String</code>、<code>&amp;str</code>、字面字符串（例如 <code>&quot;hello&quot;</code>），甚至更多。这使得它非常灵活。</li></ul><p>二者之间的区别：</p><ul><li><code>&amp;String</code> 是对 <code>String</code> 类型的引用，而 <code>&amp;str</code> 是对字符串切片的引用。</li><li><code>&amp;String</code> 只能引用 <code>String</code> 类型的数据，而 <code>&amp;str</code> 可以引用各种字符串数据。</li><li><code>&amp;String</code> 是不可变引用，不允许修改字符串数据，而 <code>&amp;str</code> 也是不可变引用，同样不允许修改字符串数据。</li></ul><p>在Rust中，通常会使用 <code>&amp;str</code> 来引用字符串数据，因为它更通用，并且可以用于多种不同的字符串类型。但在某些情况下，需要明确使用 <code>&amp;String</code>，例如，如果需要传递一个 <code>String</code> 的引用作为参数，并且不允许修改原始字符串。</p><ol><li><p><strong>String</strong>:</p><ul><li><code>String</code> 是一个可增长的、可变的、有所有权的、UTF-8 编码的字符串类型。</li><li>它是存储在堆上的，并可以调整其大小。</li><li>这是你创建和修改字符串时使用的主要类型。</li></ul><p>示例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>​</span></span>
<span class="line"><span>let mut s = String::from(&quot;hello&quot;);</span></span>
<span class="line"><span>s.push_str(&quot;, world!&quot;);  // 在字符串的末尾追加文字</span></span></code></pre></div></li><li><p><strong>str</strong>:</p><ul><li><code>str</code> 是一个不可变的固定大小的字符串切片。</li><li>它通常以其借用的形式出现，称为 <code>&amp;str</code>。</li><li>它是字符串数据的视图，而不是实际的所有者。</li></ul><p>示例：</p><div class="language- vp-adaptive-theme"><button title="Copy Code" class="copy"></button><span class="lang"></span><pre class="shiki shiki-themes github-light github-dark vp-code" tabindex="0"><code><span class="line"><span>let s: &amp;str = &quot;hello world&quot;;</span></span></code></pre></div></li></ol><p>\\</p>`,86)]))}const m=n(l,[["render",t]]);export{g as __pageData,m as default};
