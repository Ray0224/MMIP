我換兩張簡單一點的 她配對就比較正確 蛋城市為何沒把他們娉接在一起

可以。這次我幫你整理成一份完整的 Quiz 4 影像拼接程式，包含：

讀取 IMG_4.jpg、IMG_5.jpg
SIFT 特徵偵測
BFMatcher 特徵匹配
Lowe Ratio Test
顯示初步匹配結果
RANSAC 計算 Homography
顯示 RANSAC 過濾後的匹配
計算 Inlier Ratio
Perspective Transformation
將兩張圖片放到同一個畫布
顯示最終拼接結果

假設你的檔案：

MMIP/
└── week4/
    ├── data/
    │   ├── IMG_4.jpg
    │   └── IMG_5.jpg
    └── quiz4.py

程式如下：

import cv2
import numpy as np
import matplotlib.pyplot as plt


# =========================================================
# 1. 讀取兩張圖片
# =========================================================

image1 = cv2.imread("data/IMG_4.jpg")
image2 = cv2.imread("data/IMG_5.jpg")


if image1 is None:
    print("IMG_4.jpg 讀取失敗！")
    exit()


if image2 is None:
    print("IMG_5.jpg 讀取失敗！")
    exit()


print("--------------------------------")
print("圖片讀取成功")
print("--------------------------------")

print("IMG_4 尺寸：", image1.shape)
print("IMG_5 尺寸：", image2.shape)

print("--------------------------------")


# =========================================================
# 2. 建立 SIFT
# =========================================================

sift = cv2.SIFT_create()


# =========================================================
# 3. SIFT 特徵偵測
# =========================================================

keypoints1, descriptors1 = sift.detectAndCompute(
    image1,
    None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    image2,
    None
)


print("IMG_4 特徵點數量：", len(keypoints1))
print("IMG_5 特徵點數量：", len(keypoints2))

print("--------------------------------")


# =========================================================
# 4. BFMatcher
# =========================================================

bf = cv2.BFMatcher()


matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


print("原始匹配數量：", len(matches))


# =========================================================
# 5. Lowe Ratio Test
# =========================================================

good_matches = []


for m, n in matches:

    if m.distance < 0.75 * n.distance:

        good_matches.append(m)


print("Lowe Ratio Test 後：", len(good_matches))

print("--------------------------------")


# =========================================================
# 6. 顯示 Lowe Ratio Test 後的匹配
# =========================================================

match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.figure(figsize=(18, 8))

plt.imshow(
    cv2.cvtColor(
        match_image,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"Feature Matching - {len(good_matches)} matches"
)

plt.axis("off")

plt.show()


# =========================================================
# 7. 檢查匹配數量
# =========================================================

if len(good_matches) < 4:

    print("--------------------------------")
    print("匹配點不足")
    print("無法進行 Homography")
    print("--------------------------------")

    exit()


# =========================================================
# 8. 取得匹配點座標
# =========================================================

src_points = np.float32([
    keypoints1[m.queryIdx].pt
    for m in good_matches
]).reshape(-1, 1, 2)


dst_points = np.float32([
    keypoints2[m.trainIdx].pt
    for m in good_matches
]).reshape(-1, 1, 2)


# =========================================================
# 9. RANSAC + Homography
# =========================================================

H, mask = cv2.findHomography(
    src_points,
    dst_points,
    cv2.RANSAC,
    5.0
)


if H is None:

    print("--------------------------------")
    print("Homography 計算失敗")
    print("--------------------------------")

    exit()


# =========================================================
# 10. 找出 RANSAC 有效匹配
# =========================================================

inlier_matches = []


for i, m in enumerate(good_matches):

    if mask[i]:

        inlier_matches.append(m)


inlier_count = len(inlier_matches)

total_matches = len(good_matches)


inlier_ratio = (
    inlier_count /
    total_matches *
    100
)


print("--------------------------------")
print("RANSAC 結果")
print("--------------------------------")

print("Lowe Ratio Test 後：", total_matches)

print("RANSAC 有效匹配：", inlier_count)

print(
    f"Inlier Ratio：{inlier_ratio:.2f}%"
)

print("--------------------------------")


# =========================================================
# 11. 顯示 RANSAC 過濾後的匹配
# =========================================================

inlier_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


plt.figure(figsize=(18, 8))

plt.imshow(
    cv2.cvtColor(
        inlier_image,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    f"RANSAC Inlier Matches - {inlier_count} matches"
)

plt.axis("off")

plt.show()


# =========================================================
# 12. 取得圖片尺寸
# =========================================================

height1, width1 = image1.shape[:2]

height2, width2 = image2.shape[:2]


# =========================================================
# 13. IMG_4 的四個角
# =========================================================

corners1 = np.float32([
    [0, 0],
    [width1, 0],
    [width1, height1],
    [0, height1]
]).reshape(
    -1,
    1,
    2
)


# =========================================================
# 14. 使用 Homography
#     將 IMG_4 的四個角轉換到 IMG_5 座標系統
# =========================================================

transformed_corners1 = cv2.perspectiveTransform(
    corners1,
    H
)


# =========================================================
# 15. IMG_5 的四個角
# =========================================================

corners2 = np.float32([
    [0, 0],
    [width2, 0],
    [width2, height2],
    [0, height2]
]).reshape(
    -1,
    1,
    2
)


# =========================================================
# 16. 合併兩張圖片的所有角點
# =========================================================

all_corners = np.concatenate(
    (
        transformed_corners1,
        corners2
    ),
    axis=0
)


# =========================================================
# 17. 計算最小與最大座標
# =========================================================

[x_min, y_min] = np.int32(
    all_corners.min(
        axis=0
    ).ravel() - 0.5
)


[x_max, y_max] = np.int32(
    all_corners.max(
        axis=0
    ).ravel() + 0.5
)


# =========================================================
# 18. 建立平移矩陣
#
# 如果圖片有負座標，
# 把整張圖片往右 / 往下平移
# =========================================================

translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
])


# =========================================================
# 19. 計算最後畫布大小
# =========================================================

output_width = x_max - x_min

output_height = y_max - y_min


print("--------------------------------")
print("拼接畫布尺寸")
print("--------------------------------")

print(
    "Width :",
    output_width
)

print(
    "Height:",
    output_height
)

print("--------------------------------")


# =========================================================
# 20. Perspective Transformation
#
# 將 IMG_4 變形到 IMG_5 的座標系統
# =========================================================

warped_image1 = cv2.warpPerspective(
    image1,
    translation @ H,
    (
        output_width,
        output_height
    )
)


# =========================================================
# 21. 建立最終畫布
# =========================================================

result = warped_image1.copy()


# =========================================================
# 22. 放入 IMG_5
# =========================================================

x_offset = -x_min
y_offset = -y_min


result[
    y_offset :
    y_offset + height2,

    x_offset :
    x_offset + width2
] = image2


# =========================================================
# 23. 顯示最終拼接結果
# =========================================================

plt.figure(figsize=(18, 9))


plt.imshow(
    cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )
)


plt.title(
    "Image Stitching Result"
)


plt.axis("off")


plt.show()
這次你主要看這 3 個結果
① Feature Matching

你目前看到的：

Feature Matching - 386 matches

這是 Lowe Ratio Test 後的結果。

② RANSAC Inlier Matches

這個才是我們現在最關心的。

例如：

Lowe Ratio Test 後：386
RANSAC 有效匹配：320
Inlier Ratio：82.90%

如果 RANSAC 後的線大部分都集中在真正重疊的位置，代表 Homography 的估計有可靠的資料基礎。

③ Image Stitching Result

最後才是：

IMG_4 + IMG_5
        ↓
   一張完整圖片
另外提醒一點

這個版本是「先完成 Quiz 4 基礎要求」的版本，還沒有做真正的 Blending（影像融合）。

所以最後兩張圖片的重疊區域可能會看到：

明顯接縫
亮度不同
一張圖片蓋住另一張圖片

這是正常的。

等我們確認這個版本能正常找到 Homography 並完成基本拼接後，再做下一步的 影像融合與亮度調整，這會直接對應你 Quiz 4 的進階題。

有些地方拼接的完美 但有不完美的地方 然後現在程式會先顯示sift配對 關掉視窗後會出現下一個 在關掉視窗後顯示娉街的結果 改成按按鈕下一步

可以，這樣比較像一個真正的「影像拼接小工具」。

我們把流程改成同一個視窗，按「下一步」才進入下一階段：

開始
 ↓
① SIFT + 特徵匹配
 ↓ 按「下一步」
② RANSAC 過濾結果
 ↓ 按「下一步」
③ 最終拼接結果

不需要再關閉視窗。

而且我順便把最後結果的黑色空白區域裁掉，讓結果比較乾淨。

完整程式

假設：

data/
├── IMG_4.jpg
└── IMG_5.jpg

直接使用下面這份：

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


# =========================================================
# 1. 讀取圖片
# =========================================================

image1 = cv2.imread("data/IMG_4.jpg")
image2 = cv2.imread("data/IMG_5.jpg")


if image1 is None:
    print("IMG_4.jpg 讀取失敗！")
    exit()


if image2 is None:
    print("IMG_5.jpg 讀取失敗！")
    exit()


print("--------------------------------")
print("圖片讀取成功")
print("--------------------------------")
print("IMG_4 尺寸：", image1.shape)
print("IMG_5 尺寸：", image2.shape)
print("--------------------------------")


# =========================================================
# 2. SIFT 特徵偵測
# =========================================================

sift = cv2.SIFT_create()


keypoints1, descriptors1 = sift.detectAndCompute(
    image1,
    None
)


keypoints2, descriptors2 = sift.detectAndCompute(
    image2,
    None
)


print("IMG_4 特徵點數量：", len(keypoints1))
print("IMG_5 特徵點數量：", len(keypoints2))


# =========================================================
# 3. BFMatcher
# =========================================================

bf = cv2.BFMatcher()


matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


# =========================================================
# 4. Lowe Ratio Test
# =========================================================

good_matches = []


for m, n in matches:

    if m.distance < 0.75 * n.distance:
        good_matches.append(m)


print("原始匹配數量：", len(matches))
print("Lowe Ratio Test 後：", len(good_matches))


if len(good_matches) < 4:

    print("--------------------------------")
    print("匹配點不足，無法進行拼接")
    print("--------------------------------")

    exit()


# =========================================================
# 5. 第一階段：Feature Matching 圖片
# =========================================================

match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


match_image = cv2.cvtColor(
    match_image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 6. 取得匹配點
# =========================================================

src_points = np.float32([
    keypoints1[m.queryIdx].pt
    for m in good_matches
]).reshape(-1, 1, 2)


dst_points = np.float32([
    keypoints2[m.trainIdx].pt
    for m in good_matches
]).reshape(-1, 1, 2)


# =========================================================
# 7. RANSAC + Homography
# =========================================================

H, mask = cv2.findHomography(
    src_points,
    dst_points,
    cv2.RANSAC,
    5.0
)


if H is None:

    print("--------------------------------")
    print("Homography 計算失敗")
    print("--------------------------------")

    exit()


# =========================================================
# 8. 找出 RANSAC Inlier
# =========================================================

inlier_matches = []


for i, m in enumerate(good_matches):

    if mask[i]:

        inlier_matches.append(m)


inlier_count = len(inlier_matches)

total_matches = len(good_matches)

inlier_ratio = (
    inlier_count /
    total_matches *
    100
)


print("--------------------------------")
print("RANSAC 結果")
print("--------------------------------")
print("Lowe Ratio Test：", total_matches)
print("RANSAC 有效匹配：", inlier_count)
print(f"Inlier Ratio：{inlier_ratio:.2f}%")
print("--------------------------------")


# =========================================================
# 9. 第二階段：RANSAC Matching
# =========================================================

inlier_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


inlier_image = cv2.cvtColor(
    inlier_image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 10. 計算圖片尺寸
# =========================================================

height1, width1 = image1.shape[:2]

height2, width2 = image2.shape[:2]


# =========================================================
# 11. IMG_4 四個角
# =========================================================

corners1 = np.float32([
    [0, 0],
    [width1, 0],
    [width1, height1],
    [0, height1]
]).reshape(
    -1,
    1,
    2
)


# =========================================================
# 12. 將 IMG_4 的角轉換到 IMG_5 座標
# =========================================================

transformed_corners1 = cv2.perspectiveTransform(
    corners1,
    H
)


# =========================================================
# 13. IMG_5 四個角
# =========================================================

corners2 = np.float32([
    [0, 0],
    [width2, 0],
    [width2, height2],
    [0, height2]
]).reshape(
    -1,
    1,
    2
)


# =========================================================
# 14. 計算整張畫布範圍
# =========================================================

all_corners = np.concatenate(
    (
        transformed_corners1,
        corners2
    ),
    axis=0
)


[x_min, y_min] = np.int32(
    all_corners.min(
        axis=0
    ).ravel() - 0.5
)


[x_max, y_max] = np.int32(
    all_corners.max(
        axis=0
    ).ravel() + 0.5
)


# =========================================================
# 15. 平移矩陣
# =========================================================

translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
])


# =========================================================
# 16. 畫布大小
# =========================================================

output_width = x_max - x_min

output_height = y_max - y_min


# =========================================================
# 17. Perspective Transformation
# =========================================================

warped_image1 = cv2.warpPerspective(
    image1,
    translation @ H,
    (
        output_width,
        output_height
    )
)


# =========================================================
# 18. 建立遮罩
# =========================================================

mask1 = np.ones(
    (
        height1,
        width1
    ),
    dtype=np.uint8
) * 255


warped_mask1 = cv2.warpPerspective(
    mask1,
    translation @ H,
    (
        output_width,
        output_height
    )
)


# =========================================================
# 19. 建立最後結果
# =========================================================

result = warped_image1.copy()


x_offset = -x_min
y_offset = -y_min


# =========================================================
# 20. 放入 IMG_5
# =========================================================

result[
    y_offset:
    y_offset + height2,

    x_offset:
    x_offset + width2
] = image2


# =========================================================
# 21. 自動裁掉黑色區域
# =========================================================

gray_result = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2GRAY
)


# 找非黑色區域
_, crop_mask = cv2.threshold(
    gray_result,
    5,
    255,
    cv2.THRESH_BINARY
)


# 找外接矩形
coords = cv2.findNonZero(
    crop_mask
)


if coords is not None:

    x, y, w, h = cv2.boundingRect(
        coords
    )

    result = result[
        y:y+h,
        x:x+w
    ]


# =========================================================
# 22. BGR → RGB
# =========================================================

result = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 23. 建立 GUI
# =========================================================

fig, ax = plt.subplots(
    figsize=(16, 8)
)


plt.subplots_adjust(
    bottom=0.15
)


# =========================================================
# 24. 狀態
# =========================================================

stage = 0


images = [
    match_image,
    inlier_image,
    result
]


titles = [
    f"① Feature Matching - {total_matches} matches",

    f"② RANSAC Inlier Matches - "
    f"{inlier_count} matches "
    f"({inlier_ratio:.2f}%)",

    "③ Image Stitching Result"
]


# =========================================================
# 25. 顯示第一階段
# =========================================================

image_display = ax.imshow(
    images[0]
)


ax.set_title(
    titles[0],
    fontsize=16
)


ax.axis("off")


# =========================================================
# 26. 建立按鈕
# =========================================================

button_ax = plt.axes([
    0.40,
    0.03,
    0.20,
    0.07
])


next_button = Button(
    button_ax,
    "下一步"
)


# =========================================================
# 27. 下一步按鈕功能
# =========================================================

def next_stage(event):

    global stage

    # 如果還沒到最後一階段
    if stage < 2:

        stage += 1

        image_display.set_data(
            images[stage]
        )

        ax.set_title(
            titles[stage],
            fontsize=16
        )

        fig.canvas.draw_idle()


    # 已經是最後階段
    else:

        plt.close(fig)


# =========================================================
# 28. 綁定按鈕
# =========================================================

next_button.on_clicked(
    next_stage
)


# =========================================================
# 29. 顯示 GUI
# =========================================================

plt.show()
現在操作方式會變成

程式執行後直接出現第一個畫面：

① Feature Matching

例如：

Feature Matching - 386 matches

[圖片]

             ┌────────┐
             │ 下一步 │
             └────────┘

你確認完後按：

「下一步」

② RANSAC

變成：

RANSAC Inlier Matches - 131 matches (33.94%)

[圖片]

             ┌────────┐
             │ 下一步 │
             └────────┘

這時你可以清楚看到：

原本 386 個匹配，經過 RANSAC 後只留下 131 個。

這個結果其實非常適合放到你的 Quiz 4 報告裡。

③ 最終拼接

再按一次：

「下一步」

就會變成：

Image Stitching Result

[完整拼接圖片]

不需要再關閉任何視窗。

另外，你現在這組結果其實滿有價值

你現在：

386 → 131

也就是 RANSAC 保留約：

33.94%

而且從你第二張圖來看，131 條線主要集中在：

上方冷氣附近
中間長條物
下方架子

所以 Homography 確實抓到了一部分正確的幾何關係。

但最後圖片上方出現很大的黑色區域、部分物體有接縫，這跟我們前面說的 拍攝視角造成的視差（parallax） 和目前簡單的「直接覆蓋」拼接方式都有關。

下一步如果你要把 Quiz 4 做得更完整，我們可以在這個版本上加入 Blending（影像融合）＋亮度校正，這就剛好對應老師進階題的「前處理」與改善拼接效果。

這個按鈕的下一步沒有顯示出來 她顯示成字字字
然後判定的部分 衣櫃的角他沒有配對到 還有橫樑的線也沒配對到 導致拼接的結果不佳

對，你現在遇到的是兩個不同問題，而且從你貼的結果可以看得很清楚。

① 「下一步」按鈕文字顯示異常

這是 Matplotlib 中文字型問題，不是按鈕功能壞掉。把 Matplotlib 的中文字型指定成 Windows 的「微軟正黑體」即可。

② SIFT 沒有抓到衣櫃角、橫樑

這是比較重要的問題。

SIFT 不是「找物體的角」，而是找具有局部紋理、梯度變化的特徵。

所以：

衣櫃角
───────┐
       │
       │

雖然人眼很容易看出這是一個角，但如果那個位置的紋理很單純，SIFT 不一定會把它當成特徵。

橫樑也是類似：

────────────────────────

它是一條很長、很規律的線，SIFT 不一定能找到足夠穩定的特徵。

而你這張照片還有大量：

白牆
水平線
冷氣
重複紋理

所以容易出現「匹配很多，但特徵集中在某些區域」的情況。

我們先改成比較適合這個 Quiz 的版本

我會做 4 個改善：

改善 1：SIFT 增加特徵數量
sift = cv2.SIFT_create(
    nfeatures=3000
)

讓 SIFT 找更多特徵。

改善 2：加入 CLAHE

先改善局部對比：

原圖
 ↓
CLAHE
 ↓
SIFT

這對你的白牆、衣櫃、橫樑這種低對比區域有機會增加特徵。

改善 3：匹配改得更嚴格

原本：

0.75

改成：

0.70

降低錯誤匹配。

改善 4：RANSAC 後再進行拼接

而且我們會繼續保留：

Feature Matching
       ↓
RANSAC
       ↓
Image Stitching
完整修改版

這版還會修正你說的「下一步」按鈕問題。

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.font_manager import FontProperties


# =========================================================
# 1. 設定中文字型
# =========================================================

font_path = "C:/Windows/Fonts/msjh.ttc"

try:
    chinese_font = FontProperties(
        fname=font_path
    )
except:
    chinese_font = None


# =========================================================
# 2. 讀取圖片
# =========================================================

image1 = cv2.imread("data/IMG_4.jpg")
image2 = cv2.imread("data/IMG_5.jpg")


if image1 is None:
    print("IMG_4.jpg 讀取失敗！")
    exit()


if image2 is None:
    print("IMG_5.jpg 讀取失敗！")
    exit()


print("--------------------------------")
print("圖片讀取成功")
print("--------------------------------")
print("IMG_4 尺寸：", image1.shape)
print("IMG_5 尺寸：", image2.shape)
print("--------------------------------")


# =========================================================
# 3. 前處理：CLAHE
# =========================================================

def preprocess_image(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


gray1 = preprocess_image(image1)
gray2 = preprocess_image(image2)


# =========================================================
# 4. 建立 SIFT
# =========================================================

sift = cv2.SIFT_create(
    nfeatures=3000,
    contrastThreshold=0.02,
    edgeThreshold=10,
    sigma=1.6
)


# =========================================================
# 5. SIFT 特徵偵測
# =========================================================

keypoints1, descriptors1 = sift.detectAndCompute(
    gray1,
    None
)

keypoints2, descriptors2 = sift.detectAndCompute(
    gray2,
    None
)


print("IMG_4 特徵點數量：", len(keypoints1))
print("IMG_5 特徵點數量：", len(keypoints2))

print("--------------------------------")


# =========================================================
# 6. BFMatcher
# =========================================================

bf = cv2.BFMatcher(
    cv2.NORM_L2
)


matches = bf.knnMatch(
    descriptors1,
    descriptors2,
    k=2
)


# =========================================================
# 7. Lowe Ratio Test
# =========================================================

good_matches = []


for m, n in matches:

    if m.distance < 0.70 * n.distance:

        good_matches.append(m)


print("原始匹配數量：", len(matches))
print("Lowe Ratio Test 後：", len(good_matches))

print("--------------------------------")


if len(good_matches) < 4:

    print("匹配點不足，無法進行拼接")
    exit()


# =========================================================
# 8. Feature Matching 圖
# =========================================================

match_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    good_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


match_image = cv2.cvtColor(
    match_image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 9. 取得匹配座標
# =========================================================

src_points = np.float32([
    keypoints1[m.queryIdx].pt
    for m in good_matches
]).reshape(
    -1,
    1,
    2
)


dst_points = np.float32([
    keypoints2[m.trainIdx].pt
    for m in good_matches
]).reshape(
    -1,
    1,
    2
)


# =========================================================
# 10. RANSAC + Homography
# =========================================================

H, mask = cv2.findHomography(
    src_points,
    dst_points,
    cv2.RANSAC,
    4.0
)


if H is None:

    print("Homography 計算失敗")
    exit()


# =========================================================
# 11. RANSAC Inliers
# =========================================================

inlier_matches = []


for i, m in enumerate(good_matches):

    if mask[i]:

        inlier_matches.append(m)


inlier_count = len(
    inlier_matches
)

total_matches = len(
    good_matches
)


inlier_ratio = (
    inlier_count /
    total_matches *
    100
)


print("--------------------------------")
print("RANSAC 結果")
print("--------------------------------")

print(
    "Lowe Ratio Test：",
    total_matches
)

print(
    "RANSAC 有效匹配：",
    inlier_count
)

print(
    f"Inlier Ratio：{inlier_ratio:.2f}%"
)

print("--------------------------------")


# =========================================================
# 12. RANSAC 匹配圖片
# =========================================================

inlier_image = cv2.drawMatches(
    image1,
    keypoints1,
    image2,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)


inlier_image = cv2.cvtColor(
    inlier_image,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 13. 圖片尺寸
# =========================================================

height1, width1 = image1.shape[:2]

height2, width2 = image2.shape[:2]


# =========================================================
# 14. IMG_4 四個角
# =========================================================

corners1 = np.float32([
    [0, 0],
    [width1, 0],
    [width1, height1],
    [0, height1]
]).reshape(
    -1,
    1,
    2
)


# =========================================================
# 15. Perspective Transform
# =========================================================

transformed_corners1 = cv2.perspectiveTransform(
    corners1,
    H
)


# =========================================================
# 16. IMG_5 四個角
# =========================================================

corners2 = np.float32([
    [0, 0],
    [width2, 0],
    [width2, height2],
    [0, height2]
]).reshape(
    -1,
    1,
    2
)


# =========================================================
# 17. 計算畫布範圍
# =========================================================

all_corners = np.concatenate(
    (
        transformed_corners1,
        corners2
    ),
    axis=0
)


[x_min, y_min] = np.int32(
    all_corners.min(
        axis=0
    ).ravel() - 0.5
)


[x_max, y_max] = np.int32(
    all_corners.max(
        axis=0
    ).ravel() + 0.5
)


# =========================================================
# 18. 平移矩陣
# =========================================================

translation = np.array([
    [1, 0, -x_min],
    [0, 1, -y_min],
    [0, 0, 1]
])


output_width = x_max - x_min
output_height = y_max - y_min


# =========================================================
# 19. Warp IMG_4
# =========================================================

warped_image1 = cv2.warpPerspective(
    image1,
    translation @ H,
    (
        output_width,
        output_height
    )
)


# =========================================================
# 20. 建立遮罩
# =========================================================

mask1 = np.ones(
    (
        height1,
        width1
    ),
    dtype=np.uint8
) * 255


warped_mask1 = cv2.warpPerspective(
    mask1,
    translation @ H,
    (
        output_width,
        output_height
    )
)


# =========================================================
# 21. 建立結果
# =========================================================

result = warped_image1.copy()


x_offset = -x_min
y_offset = -y_min


# =========================================================
# 22. 放入 IMG_5
# =========================================================

result[
    y_offset:
    y_offset + height2,

    x_offset:
    x_offset + width2
] = image2


# =========================================================
# 23. 建立有效區域
# =========================================================

gray_result = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2GRAY
)


_, crop_mask = cv2.threshold(
    gray_result,
    5,
    255,
    cv2.THRESH_BINARY
)


coords = cv2.findNonZero(
    crop_mask
)


if coords is not None:

    x, y, w, h = cv2.boundingRect(
        coords
    )

    result = result[
        y:y+h,
        x:x+w
    ]


# =========================================================
# 24. BGR → RGB
# =========================================================

result = cv2.cvtColor(
    result,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# 25. 建立三個階段
# =========================================================

images = [
    match_image,
    inlier_image,
    result
]


titles = [
    f"① 特徵匹配 - {total_matches} matches",

    f"② RANSAC 過濾 - "
    f"{inlier_count} matches "
    f"({inlier_ratio:.2f}%)",

    "③ 影像拼接結果"
]


# =========================================================
# 26. 建立視窗
# =========================================================

fig, ax = plt.subplots(
    figsize=(16, 8)
)


plt.subplots_adjust(
    bottom=0.18
)


# =========================================================
# 27. 顯示第一張
# =========================================================

image_display = ax.imshow(
    images[0]
)


if chinese_font:

    ax.set_title(
        titles[0],
        fontsize=16,
        fontproperties=chinese_font
    )

else:

    ax.set_title(
        titles[0],
        fontsize=16
    )


ax.axis("off")


# =========================================================
# 28. 建立按鈕
# =========================================================

button_ax = plt.axes([
    0.40,
    0.035,
    0.20,
    0.08
])


next_button = Button(
    button_ax,
    "下一步"
)


# 強制指定中文字型
if chinese_font:

    next_button.label.set_fontproperties(
        chinese_font
    )

    next_button.label.set_fontsize(
        14
    )


# =========================================================
# 29. 階段控制
# =========================================================

stage = 0


# =========================================================
# 30. 下一步
# =========================================================

def next_stage(event):

    global stage

    if stage < 2:

        stage += 1

        image_display.set_data(
            images[stage]
        )

        if chinese_font:

            ax.set_title(
                titles[stage],
                fontsize=16,
                fontproperties=chinese_font
            )

        else:

            ax.set_title(
                titles[stage],
                fontsize=16
            )

        fig.canvas.draw_idle()

    else:

        plt.close(fig)


# =========================================================
# 31. 綁定按鈕
# =========================================================

next_button.on_clicked(
    next_stage
)


# =========================================================
# 32. 顯示
# =========================================================

plt.show()
這版跟上一版主要差在哪裡？
SIFT

原本：

sift = cv2.SIFT_create()

現在：

sift = cv2.SIFT_create(
    nfeatures=3000,
    contrastThreshold=0.02,
    edgeThreshold=10,
    sigma=1.6
)

讓它更積極尋找特徵。

前處理

現在多了一層：

原始圖片
   ↓
灰階
   ↓
CLAHE
   ↓
SIFT

CLAHE 是 Contrast Limited Adaptive Histogram Equalization，它會改善局部區域的對比。

這對你現在這種：

白牆
衣櫃
橫樑

低紋理、低對比區域可能比較有幫助。

Matching

從：

0.75

改成：

0.70

因此會比較嚴格，只留下更可靠的匹配。

RANSAC

從：

5.0

改成：

4.0

讓幾何誤差要求稍微嚴格一些。

但有一件事情要特別注意

即使我們這樣修改，也不能保證一定會抓到衣櫃角和橫樑。

因為這不是單純「參數不夠好」的問題。

SIFT 的本質是：

找具有辨識性的局部特徵。

它不是：

找圖片中的所有直線和角落。

所以你的：

衣櫃角
┌────
│
│

和：

橫樑
────────────────

人眼認為非常重要，但 SIFT 可能認為它們不是好的特徵。

這反而可以變成你的 Quiz 4 進階實驗

你老師要求：

嘗試在影像拼接前加入適當的前處理方法，提升特徵偵測、匹配與拼接的穩定性。

我們現在正好可以比較：

                    特徵數量
                       ↓
原始圖片 → SIFT ─────────────→ Matching → RANSAC → 拼接

                    VS

原始圖片
   ↓
CLAHE
   ↓
SIFT ───────────────────────→ Matching → RANSAC → 拼接

然後比較：

SIFT 特徵點數量
Lowe Ratio 後匹配數量
RANSAC Inlier 數量
Inlier Ratio
最終拼接結果

這樣就不是單純「程式有跑出結果」，而是真的有做老師要求的進階實驗。

另外，你現在這組照片有明顯的視差（parallax），所以即使匹配全部正確，也可能在衣櫃、橫樑等不同深度物體附近出現接縫或重影。這部分是 Homography 本身的限制，不是單純把 SIFT 調強就能完全解決。