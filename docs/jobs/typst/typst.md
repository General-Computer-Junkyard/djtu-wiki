# Typst 推荐指南

作为写论文或者排版之类的问题,LaTeX可以非常完美的解决所有论文格式的问题,但LaTeX学习曲线太陡,对于想快速排版入门的同学来说,有一定的上手难度,typst是基于rust写的新的排版工具,相对于此比较轻量和简单上手,有很好的社区讨论,可以在相对应的资源频道快速获取公式符号和模板上手使用

## 推荐自学资料

[The Raindrop-Blue Book (Typst中文教程)](https://typst-doc-cn.github.io/tutorial/)

[Tinymist Docs(英文版)](https://myriad-dreamin.github.io/tinymist/introduction.html)

[模板](https://typst.app/universe/search/)

[数学表达式照片识别](https://github.com/ParaN3xus/typress)

[手写识别数字符号](https://detypify.quarticcat.com/)

## 为什么推荐 Typst？

1. **上手简单**  
    LaTeX 的学习曲线太陡，担心代码写得不够简洁。typst的语法非常接近 Markdown，但功能却强大得多。
    
    例如，你可以轻松实现标题、段落、加粗、斜体、图片、公式这些排版需求，而不用像 LaTeX 那样背一堆复杂命令。

2. **实时预览超方便**  
   我想大家可能都有这样的经验——在 LaTeX 里调个格式要编译无数次，而 Typst 提供了实时预览功能，你改动一点就能立刻看到结果，减少了编译等待的时间，节省了不少精力。

3. **强大的数学公式支持**  
   对我们学工科或者理科的同学来说，公式排版是绕不开的事。Typst 在这方面真的没得说，语法跟 LaTeX 差不多，但用起来更直观，还不容易出错。你可以直接像这样写公式：

   ```typst
   $E = mc^2$
   ```

   简单明了。

4. **表格、图表一站式搞定**  
   如果你要插入表格或者图表，Typst 的语法也很简洁。比如表格只需要几行代码，就可以自动排好格式，不用像 Word 那样反复调整。

   ```typst
   #table(
     [项目, 时间, 结果],
     [A, 2022, 通过],
     [B, 2023, 进行中]
   )
   ```

   调整起来比 Excel 还轻松，视觉效果也很专业。

5. **定制灵活，适合复杂项目**  
   如果你要写论文，甚至是毕业论文这种需要统一排版格式的长文档，Typst 的灵活性也能满足你。你可以定义模板，甚至封装自己的一些样式。一次设置，全文调用，格式高度统一。

## 笔者体验

刚开始用 Typst 的时候，我其实有点担心——担心它会不会功能不够，担心它的语法是不是太简单，做不到我想要的复杂效果。结果用下来发现，这些担心完全是多余的。它不仅能满足日常作业、论文排版的需要，甚至在做一些带有公式、表格的复杂文档时，它也完全不逊色于 LaTeX，反而更快更省心。

Typst 是一款既现代又高效的排版工具，特别适合想要高效完成文档的你们。它介于 Word 和 LaTeX 之间，结合了两者的优点，没有 LaTeX 那么陡峭的学习曲线，也没有 Word 的局限性。