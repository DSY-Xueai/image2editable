# 适用于网页版的图片转可编辑PPT转换方法

原理：将每张图片的视觉元素拆分成独立的图片，文字拆分成可编辑文本框

以下示例使用的是ChatGPT网页版，模型选择 5.6 High

耗时与图片复杂度有关，测试了很多张，基本都在10分钟内

保姆级教程在网页转换示例下方

仓库右上角点个star，感谢支持哟

## 网页转换示例

|                             原图                             |                       转换后可编辑效果                       |                          耗时与过程                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![image-20260907220124342](docs/images/image-20260907220124342.png) | ![image-20260907220147484](docs/images/image-20260907220147484.png) | ![image-20260907215947690](docs/images/image-20260907215947690.png) |
| ![image-20260907220259874](docs/images/image-20260907220259874.png) | ![image-20260907220244583](docs/images/image-20260907220244583.png) | ![image-20260907220210203](docs/images/image-20260907220210203.png) |
| ![image-20260907220644020](docs/images/image-20260907220644020.png) | ![image-20260907220945352](docs/images/image-20260907220945352.png) | ![image-20260907221002947](docs/images/image-20260907221002947.png) |

------

## 保姆教程

![image-20260907222207460](docs/images/image-20260907222207460.png)

![image-20260907223306672](docs/images/image-20260907223306672.png)

**提示词下面还有些说明，建议看完**

------

```markdown
本项目用于将当前聊天直接上传的单张图片拆分为独立视觉素材，并使用拆分结果自动重建可编辑 PPTX。

新聊天中仅上传 1 张图片时，直接执行完整流程：

阶段 1：拆分全部非文字视觉元素
阶段 2：使用全部拆分素材重建可编辑 PPT

不进行确认，不要求重复输入提示词，不等待“继续”，直接执行。

==================================================
一、任务隔离
==================================================

每个聊天只处理当前聊天直接上传的源图片。

禁止使用、引用、混入或复用其他聊天中的：

- 源图片
- PNG 素材
- PPT
- 任何视觉元素

1 张源图片对应 1 个独立聊天任务。

==================================================
二、阶段 1：视觉元素拆分
==================================================

将源图片中的全部非文字视觉元素完整拆分为独立 PNG。

必须满足以下要求：

1. 必须完成全部拆分。

持续处理当前图片中的全部视觉元素，直到全部完成。

单批处理结束不代表任务结束。

不得：
- 中途暂停等待“继续”
- 要求选择元素
- 要求选择批次
- 因元素数量较多提前结束

2. 每一个独立视觉元素分别输出为一张 PNG。

一张 PNG 只能包含一个独立视觉元素。

禁止：
- 两个或多个独立元素合并
- 多个人物合并
- 多个物品合并
- 元素与背景合并
- 拼图
- 九宫格
- Sprite Sheet
- 素材合集
- 任何形式的多元素合并输出

以“完整且具有独立素材意义的视觉对象”作为拆分单位。

不得为了减少素材数量而扩大拆分范围，也不得把一个完整对象无意义地拆成碎片。

3. 每张 PNG 必须使用与源图片完全相同的画布尺寸。

保持元素在源图中的：

- 原始坐标
- 原始尺寸
- 原始比例
- 原始方向

禁止：
- 自动裁剪画布
- 移动元素到画布中央
- 改变元素尺寸
- 拉伸
- 改变比例
- 改变方向

除当前元素外，其余区域全部透明。

4. PNG 必须是真实透明背景。

不得包含：

- 原图背景
- 白底
- 色块
- 其他元素
- 其他元素残影
- 多余背景区域
- 污染像素

不得使用矩形截图代替真正的独立透明素材。

5. 每个视觉元素必须完整、清晰。

不得出现：

- 缺边
- 截断
- 缺失
- 残影
- 重复
- 粘连
- 错误透明
- 明显锯齿
- 不必要的清晰度下降

拆分不得改变元素原本的：

- 外观
- 颜色
- 纹理
- 比例
- 风格
- 结构

6. 所有非文字视觉元素均需处理。

不依赖固定类别判断拆分范围。

当前图片中所有具有独立视觉素材意义的非文字对象，都必须纳入拆分。

7. 文字不作为 PNG 素材输出。

标题、正文、数字、标签、说明文字等普通排版文字全部留到阶段 2 重建为可编辑文本。

视觉图形与文字组合出现时，应区分图形与文字：

图形作为视觉素材处理；
文字作为可编辑文本处理。

不得因为文字位于图形内部，就把文字重复保留在最终视觉素材中。

==================================================
三、阶段 1 完整性检查
==================================================

拆分完成前必须重新对照源图片检查全部结果。

检查：

1. 是否仍存在未拆分视觉元素；
2. 是否错误合并多个独立元素；
3. 是否有元素被重复输出；
4. 是否存在缺边、截断或遗漏；
5. 是否存在背景或其他元素残影；
6. 是否真正透明；
7. 是否保持原始坐标、尺寸、比例和方向；
8. 是否错误输出普通文字；
9. 是否存在明显质量下降。

发现问题必须自行重新处理对应元素。

只有源图片中的全部非文字视觉元素均已正确拆分后，阶段 1 才算完成。

==================================================
四、阶段 1 输出
==================================================

必须直接逐张输出独立 PNG。

禁止：

- ZIP
- 压缩包
- 文件夹
- 一个总下载链接代替独立 PNG
- 拼图
- 合并素材
- 素材合集

不要输出：

- 拆分计划
- 元素清单
- 工作步骤
- 冗余说明

直接处理并输出结果。

==================================================
五、阶段 2：自动重建可编辑 PPT
==================================================

阶段 1 全部完成后，立即继续重建 PPT，不等待新的指令。

PPT 中的实际非文字视觉素材只能使用阶段 1 已经拆分得到的独立 PNG。

源图片仅用于：

- 读取文字
- 确认原始布局
- 确认对象层级
- 最终还原校验

源图片禁止直接作为 PPT 中的图片对象使用。

禁止：

- 将源图片整页插入 PPT
- 将源图片作为背景兜底
- 从源图片重新裁剪大块区域代替已拆分 PNG
- 用源图片覆盖或重复已经拆分的元素

==================================================
六、PNG 重建规则
==================================================

阶段 1 输出的全部独立 PNG 都必须用于 PPT 重建，不得遗漏。

每一个独立视觉元素在 PPT 中必须仍然是一个独立图片对象。

禁止：

- 将多个 PNG 合并后插入
- 把多个元素重新制作成一张大图片
- 使用整页图片代替独立素材
- 保留无意义的大面积透明区域作为 PPT 对象范围

阶段 1 的 PNG 使用完整源图画布，仅用于保留准确的原始坐标。

进入 PPT 重建时，必须对每一个透明 PNG 单独计算实际非透明像素边界。

根据该非透明边界：

1. 去除 PNG 四周无内容的透明区域；
2. 生成仅包含当前视觉元素实际边界的紧边界透明 PNG；
3. 保持元素本身尺寸、比例、方向和像素内容不变；
4. 根据裁剪前的非透明边界坐标，将该元素放置到 PPT 中对应的原始位置。

PPT 中每个视觉元素图片对象的：

- X 坐标
- Y 坐标
- 宽度
- 高度

必须由该元素在源图片中的实际非透明边界换算得到。

换算关系：

PPT_X = 元素左边界X / 源图片宽度 × PPT页面宽度
PPT_Y = 元素上边界Y / 源图片高度 × PPT页面高度
PPT_Width = 元素边界宽度 / 源图片宽度 × PPT页面宽度
PPT_Height = 元素边界高度 / 源图片高度 × PPT页面高度

禁止通过视觉估算重新摆放元素。

禁止为了缩小框选范围而改变元素原始大小或比例。

最终要求：

在 PowerPoint 中单击任意视觉元素时，其图片框选范围应贴合该视觉元素自身的实际边界，而不是整张幻灯片大小。

透明边距应尽可能小，但不得裁掉元素真实边缘、阴影、半透明边缘或其他属于该元素的有效像素。

元素的前后层级按照源图片恢复。

==================================================
七、文字重建
==================================================

源图片中的普通排版文字必须重新建立为独立可编辑文本框。

不得把普通文字制作成图片。

不得遗漏文字。

文字重建不仅要保证可编辑，还必须尽可能保持与原图一致的视觉样式。

必须尽可能还原：

- 文字内容
- 字体
- 字号
- 字重
- 字形风格
- 颜色
- 对齐
- 行距
- 字距
- 位置
- 方向
- 旋转角度
- 填充颜色
- 多色文字效果
- 渐变效果
- 描边
- 阴影
- 发光
- 透明度
- 圆角字、手写字、卡通字、艺术字等原始视觉风格

文字的实际宽度、高度、换行位置和整体占用范围必须尽可能与原图一致。

不得因为字体替换或字号估算导致：

- 文字框明显变宽或变窄
- 文字位置偏移
- 换行位置改变
- 字符挤压
- 字符间距明显变化
- 标题比例明显失真

对于彩色字体、渐变字体、描边字体、阴影字体、艺术字、标题字、特殊装饰字体，不得简化成普通单色默认字体。

原图文字存在多个颜色、多个样式或不同字重时，应在同一文本框内按对应字符或文字区段分别还原，不得统一成单一格式。

原图中同一行或同一文本框内存在不同颜色、字号、字重、字体或其他效果时，应保留这些局部差异。

必须优先使用与原图视觉最一致的字体和样式。

不得擅自替换为与原图差异明显的通用默认字体。

无法完全确认字体名称时，应根据原图中文字的字形特征选择视觉效果最接近的可用字体，并继续通过字号、字重、字距、缩放、描边、阴影等属性还原原始效果。

不得仅因为无法确定准确字体名称，就直接使用统一默认字体。

文字效果必须尽可能通过 PowerPoint 可编辑文字属性实现。

不得为了还原特殊字体效果而直接把整段普通排版文字转换为图片。

已经建立为文本框的文字，不得在 PNG 或其他图片对象中再次重复出现。

最终必须能够直接修改文字内容，并在修改前保持与原图文字视觉效果尽可能一致。

==================================================
八、PPT 最终结构
==================================================

最终 PPT 页面应由以下独立对象组成：

- 阶段 1 拆分得到的独立 PNG
- 独立可编辑文本框

所有 PNG 可以：

- 单独选中
- 单独移动
- 单独缩放
- 单独删除
- 调整前后层级

文字可以直接编辑。

禁止使用“整页原图 + 文本框”的方式伪装成可编辑 PPT。

禁止使用“大块裁剪图片 + 文本框”的方式降低拆分标准。

==================================================
九、最终检查
==================================================

正式输出 PPT 前检查：

1. 阶段 1 的全部 PNG 是否均已使用；
2. 是否遗漏视觉元素；
3. 是否有多个元素被重新合并；
4. 是否错误插入源图片整页；
5. PNG 位置、尺寸、比例是否正确；
6. PNG 前后层级是否正确；
7. 普通文字是否为独立可编辑文本框；
8. 是否存在文字重复；
9. 页面视觉效果是否与源图片一致；
10. PPT 是否可以正常打开和编辑；
11. 每个 PNG 图片对象的框选范围是否贴合自身视觉元素；
12. 是否存在因为完整画布透明区域导致图片框选范围接近整张幻灯片；
13. 是否在缩小透明边距过程中裁掉了元素真实边缘、阴影或半透明像素；
14. 文字的字体、字号、字重、颜色、字距、行距、位置、方向和整体占用范围是否与原图一致；
15. 彩色字体、渐变字体、描边字体、阴影字体、艺术字、标题字等特殊样式是否被错误简化为普通默认字体；
16. 同一文本框内部存在多种颜色、字号、字体、字重或其他局部样式时，是否保留了对应差异；
17. 文本框是否因字体替换导致明显错位、换行变化、尺寸异常或比例失真；
18. 所有可编辑文本框是否在保持可编辑的同时，视觉样式尽可能接近原图。

发现问题先修复，再输出。

==================================================
十、最终输出
==================================================

最终必须实际创建并输出一个 `.pptx` 文件。

不能只：

- 描述已经完成
- 提供代码
- 提供转换说明

必须交付真正可以打开和编辑的 PPTX 文件。
```

------

![image-20260907224907090](docs/images/image-20260907224907090.png)

![image-20260907224551544](docs/images/image-20260907224551544.png)

### 常问问题：

1. **为什么只建议一张图片一个 Chat？**
    因为要求“完整拆分所有元素 + 后续精准重建 PPT”。一张图一个 Chat 能避免不同页面的素材、坐标、文字、图层互相混淆，稳定性最高。
2. **为什么不建议一次发多张？**
    不是绝对不能，而是一次多张会同时带来：

​	1.元素数量暴增，容易漏拆或合并错元素；

​	2.不同图片的素材容易串页；

​	3.单次生成/处理数量存在限制；

​	4.后续很难保证“哪个 PNG 属于哪一页、哪个坐标”。

# English Version

## Method for Converting Images into Editable PowerPoint Presentations on the Web

Principle: split every independent non-text visual element in the source image into its own PNG asset, then rebuild all ordinary text as editable text boxes.

This workflow is for the ChatGPT web interface. Select the 5.6 High model. Processing time depends on image complexity; current tests generally finish within 10 minutes. The tutorial appears below the web conversion examples.

## Web Conversion Examples

| Original image | Editable result | Time and process |
| :---: | :---: | :---: |
| ![image-20260907220124342](docs/images/image-20260907220124342.png) | ![image-20260907220147484](docs/images/image-20260907220147484.png) | ![image-20260907215947690](docs/images/image-20260907215947690.png) |
| ![image-20260907220259874](docs/images/image-20260907220259874.png) | ![image-20260907220244583](docs/images/image-20260907220244583.png) | ![image-20260907220210203](docs/images/image-20260907220210203.png) |
| ![image-20260907220644020](docs/images/image-20260907220644020.png) | ![image-20260907220945352](docs/images/image-20260907220945352.png) | ![image-20260907221002947](docs/images/image-20260907221002947.png) |
------

## Tutorial

![image-20260907222207460](docs/images/image-20260907222207460.png)

![image-20260907223306672](docs/images/image-20260907223306672.png)

**The notes below the prompt are also important; please read them.**

------

## Complete Prompt

```markdown
This project converts one source image uploaded directly in the current chat into independent visual assets and uses those assets to automatically rebuild an editable PPTX.

When a new chat contains exactly one uploaded image, execute the complete workflow immediately:

Stage 1: separate every non-text visual element.
Stage 2: use every separated asset to rebuild an editable PowerPoint presentation.

Do not ask for confirmation, request the prompt again, or wait for “continue”.

==================================================
I. TASK ISOLATION
==================================================

Process only the source image uploaded directly in the current chat. Never use, reference, mix, or reuse source images, PNG assets, PPT files, or visual elements from another chat. One source image is one independent chat task.

==================================================
II. STAGE 1: SEPARATE VISUAL ELEMENTS
==================================================

Fully separate every non-text visual element in the source image into an independent PNG.

1. Complete the entire separation. Continue through all batches until every element is finished. Never pause for “continue”, ask me to choose elements or batches, or stop early because there are many elements.

2. Output one PNG for each independent visual element. Do not merge independent elements, people, objects, the background, or create collages, grids, sprite sheets, or asset collections. Use a complete visual object with independent asset meaning as the separation unit; do not split an intact object into meaningless fragments.

3. Every PNG must use exactly the source image’s canvas size and preserve the element’s original coordinates, dimensions, proportions, and orientation. All other areas must be transparent. Do not crop the canvas, center, resize, stretch, distort, or rotate the element.

4. PNGs must have true transparency. Do not include the original background, white or colored fills, other elements, residual pixels, or rectangular screenshots.

5. Keep every element complete and clear: no missing edges, truncation, omissions, artifacts, duplication, unwanted connections, incorrect transparency, obvious aliasing, or unnecessary quality loss. Preserve its appearance, color, texture, proportions, style, and structure.

6. Process every non-text object with independent visual meaning; do not rely on a fixed category list.

7. Do not output ordinary text as PNG assets. Leave titles, body text, numbers, labels, and captions for Stage 2 as editable text. When graphics and text overlap, separate the graphic as an asset and the text as editable content; never retain the text inside the final visual asset.

==================================================
III. STAGE 1 COMPLETENESS CHECK
==================================================

Before finishing Stage 1, compare every result with the source. Check for unseparated elements, accidental merges, duplicates, missing edges or objects, background remnants, real transparency, preserved coordinates/dimensions/proportions/orientation, wrongly exported text, and quality loss. Fix any issue yourself before continuing. Stage 1 is complete only when every non-text visual element is correctly separated.

==================================================
IV. STAGE 1 OUTPUT
==================================================

Output each independent PNG directly, one by one. Do not output a ZIP, archive, folder, combined download link, collage, merged asset, asset collection, separation plan, element list, work steps, or redundant explanation.

==================================================
V. STAGE 2: AUTOMATIC EDITABLE PPT REBUILD
==================================================

Immediately rebuild the PPT after Stage 1; do not wait for another instruction. Use only the independent PNGs produced in Stage 1 for non-text visual content. Use the source image only to read text, confirm layout and layer order, and perform the final visual check. Never insert the source image as a full-page picture or background, crop large regions from it, or use it to cover or duplicate separated elements.

==================================================
VI. PNG REBUILD RULES
==================================================

Use every Stage 1 PNG exactly once as its own picture object. Do not merge PNGs or replace them with a full-page image. Before placing each asset, calculate its actual non-transparent pixel bounds, trim only the transparent margin, preserve the element’s pixels, dimensions, proportions, and orientation, and place it at its original position.

Use these conversions:

PPT_X = element left boundary X / source image width × PPT page width
PPT_Y = element top boundary Y / source image height × PPT page height
PPT_Width = element boundary width / source image width × PPT page width
PPT_Height = element boundary height / source image height × PPT page height

Do not estimate positions visually or change the original size or proportion. Each picture’s selection box must fit the element’s actual bounds, with transparent margins as small as possible without cutting real edges, shadows, semi-transparent pixels, or other valid content. Restore the original front-to-back order.

==================================================
VII. TEXT REBUILD
==================================================

Recreate all ordinary text as independent editable text boxes; never render it as an image or omit it. Match the source as closely as possible: content, font, size, weight, style, color, alignment, line spacing, character spacing, position, direction, rotation, fill, multicolor or gradient effects, outline, shadow, glow, transparency, rounded, handwritten, cartoon, and artistic styles.

Match the actual width, height, wrapping, and occupied area. Do not let font substitution or guessed sizing cause obvious shifts, changed line breaks, compressed characters, spacing changes, or distorted title proportions. Preserve local differences in color, font, size, weight, and effects within one text box. Prefer the closest available font and reproduce effects with editable PowerPoint properties. Never duplicate text in PNGs or other image objects.

==================================================
VIII. FINAL PPT STRUCTURE
==================================================

The page must consist only of the independent Stage 1 PNG picture objects and independent editable text boxes. Every PNG must be selectable, movable, resizable, deletable, and reorderable. Text must be directly editable. Do not use a full-page source image or a large cropped image as a substitute for separation.

==================================================
IX. FINAL CHECK
==================================================

Before delivery, verify that all PNGs are used, no visual elements are missing or merged, the source image is not inserted, positions, sizes, proportions, and layer order are correct, all ordinary text is editable and not duplicated, the page visually matches the source, the PPTX opens and edits normally, picture selection boxes fit their elements, transparent margins have not expanded to page size or cut valid pixels, and all text typography and local styles remain as close to the source as possible. Fix every issue before output.

==================================================
X. FINAL OUTPUT
==================================================

Actually create and output an editable `.pptx` file. Do not only describe the result, provide code, or explain the conversion.
```

------

![image-20260907224907090](docs/images/image-20260907224907090.png)

![image-20260907224551544](docs/images/image-20260907224551544.png)

### FAQ

1. **Why is one image per chat recommended?** Because complete element separation and precise PPT reconstruction are required. One image per chat prevents assets, coordinates, text, and layers from different pages from being mixed.
2. **Why not send multiple images at once?** Multiple images increase the number of elements, make omissions and incorrect merges more likely, can mix assets between pages, encounter generation limits, and make it difficult to track which PNG belongs to which page and coordinate system.