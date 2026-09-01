适用于网页版的图片转可编辑PPT提示词

原理：将每张图片的视觉元素拆分成独立的图片，文字拆分成可编辑文本框

以下示例使用网页ChatGPT 5.5 High(耗时约30分钟，与图片复杂度有关)，只针对图片，没试过PDF,图片版PPT的转换效果

目前实测下来下列Prompt效果最好的一版，实践中可按步发给AI，合并Prompt发现效果下降

## 网页转换效果示例：

|                             原图                             |                       转换后可编辑效果                       |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![Original image](docs/images/webpage-chunking-source.png) | ![Editable result](docs/images/webpage-chunking-result.png) |

------

## Prompt 1:

请将这张图片中的所有视觉元素完整拆分为独立 PNG 素材。

要求：

1. **必须自动完成全部拆分。若单次生成数量存在上限，必须自行连续分批处理，直到所有元素全部输出完毕。禁止因为单次数量限制而暂停、询问我、要求我选择元素或等待我回复。** 
2.  每一个独立视觉元素分别输出为一张 PNG，**禁止将两个或多个元素合并到同一张 PNG 中。** 
3. **每张 PNG 使用与原始图片完全相同的画布尺寸**，元素保持其在原图中的**原始坐标、原始尺寸、原始比例和方向**，其余区域全部透明。 
4.  PNG 必须为**透明背景**，不得包含原图背景、白底、色块或其他元素残影。 
5.  每个元素必须完整，不得出现缺边、截断、残影、重复、粘连或遗漏。 
6.  正式输出前自行检查全部拆分结果，发现不完整时自行重新处理，不要询问我。 
7. **所有视觉元素均需拆分，包括人物、人物配饰、图标、装饰物、气泡/图形、树木、植物及其他独立插画元素。** 
8. **不要输出任何文字内容，原图中的所有文字均忽略，不需要拆分。** 
9. **整个任务过程中不要向我确认、提问、让我选择批次或让我回复“继续”。完成一批后立即自动处理下一批。** 
10.  直接逐张输出 PNG 图片，不要 ZIP，不要打包成文件夹，不要额外输出说明文字。 
11. **只有确认所有非文字视觉元素均已拆分并输出后，任务才算完成。**

------

## Prompt 2 :

请使用刚才拆分得到的所有PNG 素材，重新还原一套完整的PPT文件。
要求:
1.按照原始页面布局，将所有 PNG 素材放回对应位置。
2.所有图片保持原比例和原位置关系，不要擅自移动、拉伸或修改。
3.在此基础上，加入可编辑文本框。
4.文本必须是独立可编辑内容，不要做成图片。
5.整体输出为一个可编辑的PPT 文件，格式为.pptx。
6.最终文件需要可以直接修改文字、调整布局。

------

# English Version

Prompt for converting images into editable PowerPoint presentations on the web

Principle: Separate the original visual elements in each image into independent images, and convert the text into editable text boxes.

The example below uses ChatGPT 5.5 High on the web (taking about 30 minutes, depending on image complexity). It has only been tested with images; the conversion results for PDFs and image-based PowerPoint files have not been tested.

Based on current testing, the prompt set below has produced the best results so far. In practice, send them to the AI one step at a time; combining them into a single prompt reduced the quality.

## Web Conversion Example:

|                        Original Image                        |                    Editable Result After Conversion                    |
| :----------------------------------------------------------: | :--------------------------------------------------------------------: |
| ![Original image](docs/images/webpage-chunking-source.png) | ![Editable result](docs/images/webpage-chunking-result.png) |

------

## Prompt 1:

Please fully separate all visual elements in this image into individual PNG assets.

Requirements:

1. **You must automatically complete the entire separation process. If there is a limit on the number of images that can be generated at one time, you must continue processing consecutive batches on your own until all elements have been output. Do not pause, ask me questions, ask me to select elements, or wait for my reply because of a per-generation quantity limit.**
2. Output each independent visual element as a separate PNG. **Do not combine two or more elements into the same PNG.**
3. **Use a canvas exactly the same size as the original image for every PNG.** Preserve the element's **original coordinates, original dimensions, original proportions, and orientation** from the original image, and make all remaining areas fully transparent.
4. Each PNG must have a **transparent background** and must not contain the original background, a white background, color blocks, or remnants of other elements.
5. Every element must be complete, with no missing edges, truncation, residual artifacts, duplication, unwanted connections, or omissions.
6. Before producing the final output, inspect all separation results yourself. If anything is incomplete, reprocess it yourself without asking me.
7. **All visual elements must be separated, including people, people's accessories, icons, decorations, bubbles/shapes, trees, plants, and any other independent illustrated elements.**
8. **Do not output any text content. Ignore all text in the original image; it does not need to be separated.**
9. **Throughout the entire task, do not ask me for confirmation, ask questions, ask me to select batches, or ask me to reply “continue.” After completing one batch, immediately process the next batch automatically.**
10. Output the PNG images directly, one by one. Do not create a ZIP file, do not package them into a folder, and do not output any additional explanatory text.
11. **The task is complete only after you have confirmed that all non-text visual elements have been separated and output.**

------

## Prompt 2:

Please use all the PNG assets separated in the previous step to recreate a complete PowerPoint file.
Requirements:
1. Place every PNG asset back in its corresponding position according to the original page layout.
2. Preserve the original proportions and positional relationships of all images. Do not move, stretch, or modify them without permission.
3. Add editable text boxes on this basis.
4. All text must remain independently editable and must not be converted into images.
5. Output the complete presentation as an editable `.pptx` file.
6. The final file must allow direct text editing and layout adjustments.
