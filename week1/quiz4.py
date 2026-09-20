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