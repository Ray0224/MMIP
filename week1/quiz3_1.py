import cv2
import numpy as np


# =========================================================
# 1. 讀取圖片
# =========================================================

image_path = "data/IMG_3.jpg"

original_image = cv2.imread(image_path)

if original_image is None:
    print("圖片讀取失敗！")
    print("請確認圖片路徑：", image_path)
    exit()


# =========================================================
# 2. 圖片顯示尺寸
# =========================================================

max_display_width = 1200
max_display_height = 800

original_height, original_width = original_image.shape[:2]

scale = min(
    max_display_width / original_width,
    max_display_height / original_height,
    1
)

display_width = int(original_width * scale)
display_height = int(original_height * scale)

display_image = cv2.resize(
    original_image,
    (display_width, display_height)
)


# =========================================================
# 3. 全域變數
# =========================================================

points = []

auto_points = None

mode = "auto"

show_image = display_image.copy()


# =========================================================
# 4. 自動偵測卡片四個角
# =========================================================

def detect_card_corners(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # 降低雜訊
    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # 邊緣偵測
    edges = cv2.Canny(
        blur,
        50,
        150
    )

    # 找輪廓
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = image.shape[0] * image.shape[1]

    candidate = None
    candidate_area = 0

    # 從最大的輪廓開始找
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:

        area = cv2.contourArea(contour)

        # 太小的物體忽略
        if area < image_area * 0.05:
            continue

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # 找四邊形
        if len(approx) == 4:

            # 必須是凸四邊形
            if cv2.isContourConvex(approx):

                if area > candidate_area:

                    candidate = approx
                    candidate_area = area


    if candidate is None:
        return None

    corners = candidate.reshape(
        4,
        2
    ).astype(np.float32)

    return order_points(corners)


# =========================================================
# 5. 四個點排序
# =========================================================

def order_points(points):

    points = np.array(
        points,
        dtype=np.float32
    )

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32
    )

    # 左上 + 右下
    sums = points.sum(axis=1)

    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]

    # 右上 + 左下
    differences = points[:, 0] - points[:, 1]

    ordered[1] = points[
        np.argmax(differences)
    ]

    ordered[3] = points[
        np.argmin(differences)
    ]

    return ordered


# =========================================================
# 6. 顯示四個角
# =========================================================

def draw_points(image, points):

    result = image.copy()

    names = [
        "TL",
        "TR",
        "BR",
        "BL"
    ]

    for i, point in enumerate(points):

        x = int(point[0] * scale)
        y = int(point[1] * scale)

        cv2.circle(
            result,
            (x, y),
            10,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            result,
            names[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    return result


# =========================================================
# 7. 滑鼠事件
# =========================================================

def mouse_callback(event, x, y, flags, param):

    global points
    global show_image

    # 只有手動模式才能點
    if mode != "manual":
        return

    if event == cv2.EVENT_LBUTTONDOWN:

        if len(points) < 4:

            # 顯示座標 → 原始圖片座標
            original_x = int(x / scale)
            original_y = int(y / scale)

            points.append([
                original_x,
                original_y
            ])

            print(
                f"手動選擇第 {len(points)} 個點："
                f"({original_x}, {original_y})"
            )

            cv2.circle(
                show_image,
                (x, y),
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                show_image,
                str(len(points)),
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.imshow(
                "Perspective Correction",
                show_image
            )


# =========================================================
# 8. 建立視窗
# =========================================================

window_name = "Perspective Correction"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_AUTOSIZE
)

cv2.setMouseCallback(
    window_name,
    mouse_callback
)


# =========================================================
# 9. 自動偵測
# =========================================================

auto_points = detect_card_corners(
    original_image
)


if auto_points is not None:

    print("--------------------------------")
    print("自動偵測成功！")
    print("--------------------------------")

    print("左上：", auto_points[0])
    print("右上：", auto_points[1])
    print("右下：", auto_points[2])
    print("左下：", auto_points[3])

    show_image = draw_points(
        display_image,
        auto_points
    )

else:

    print("--------------------------------")
    print("自動偵測失敗")
    print("請切換成手動選點")
    print("--------------------------------")


# =========================================================
# 10. 顯示操作說明
# =========================================================

print("--------------------------------")
print("操作方式")
print("--------------------------------")
print("A：使用自動偵測結果")
print("M：切換成手動選點")
print("C：開始透視校正")
print("R：重新自動偵測")
print("Q：離開")
print("--------------------------------")


cv2.imshow(
    window_name,
    show_image
)


# =========================================================
# 11. 主迴圈
# =========================================================

while True:

    key = cv2.waitKey(1) & 0xFF


    # -----------------------------------------------------
    # A：使用自動偵測
    # -----------------------------------------------------

    if key == ord('a'):

        if auto_points is not None:

            points = auto_points.copy()

            mode = "auto"

            show_image = draw_points(
                display_image,
                points
            )

            cv2.imshow(
                window_name,
                show_image
            )

            print("已使用自動偵測的四個角")


        else:

            print("目前沒有自動偵測結果")


    # -----------------------------------------------------
    # M：切換手動模式
    # -----------------------------------------------------

    elif key == ord('m'):

        mode = "manual"

        points = []

        show_image = display_image.copy()

        cv2.imshow(
            window_name,
            show_image
        )

        print("--------------------------------")
        print("已切換成手動選點")
        print("請點選卡片四個角")
        print("四個點可以任意順序")
        print("--------------------------------")


    # -----------------------------------------------------
    # R：重新自動偵測
    # -----------------------------------------------------

    elif key == ord('r'):

        auto_points = detect_card_corners(
            original_image
        )

        if auto_points is not None:

            points = auto_points.copy()

            mode = "auto"

            show_image = draw_points(
                display_image,
                auto_points
            )

            cv2.imshow(
                window_name,
                show_image
            )

            print("重新偵測成功")

        else:

            print("重新偵測失敗")


    # -----------------------------------------------------
    # C：開始校正
    # -----------------------------------------------------

    elif key == ord('c'):

        if len(points) != 4:

            print("--------------------------------")
            print("目前沒有完整的四個點")
            print("請先使用自動偵測或手動選點")
            print("--------------------------------")

            continue


        # =================================================
        # 四個點排序
        # =================================================

        ordered_points = order_points(
            points
        )


        # =================================================
        # 計算輸出尺寸
        # =================================================

        top_width = np.linalg.norm(
            ordered_points[1] -
            ordered_points[0]
        )

        bottom_width = np.linalg.norm(
            ordered_points[2] -
            ordered_points[3]
        )

        width = int(
            (top_width + bottom_width) / 2
        )


        left_height = np.linalg.norm(
            ordered_points[3] -
            ordered_points[0]
        )

        right_height = np.linalg.norm(
            ordered_points[2] -
            ordered_points[1]
        )

        height = int(
            (left_height + right_height) / 2
        )


        # =================================================
        # Perspective Transformation
        # =================================================

        src = ordered_points

        dst = np.float32([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ])


        M = cv2.getPerspectiveTransform(
            src,
            dst
        )


        result = cv2.warpPerspective(
            original_image,
            M,
            (width, height)
        )


        # =================================================
        # 限制結果顯示大小
        # =================================================

        max_result_width = 1280
        max_result_height = 720

        result_height, result_width = result.shape[:2]

        result_scale = min(
            max_result_width / result_width,
            max_result_height / result_height,
            1
        )


        if result_scale < 1:

            new_width = int(
                result_width * result_scale
            )

            new_height = int(
                result_height * result_scale
            )

            display_result = cv2.resize(
                result,
                (new_width, new_height)
            )

        else:

            display_result = result


        # =================================================
        # 顯示結果
        # =================================================

        cv2.imshow(
            "Perspective Result",
            display_result
        )

        print("--------------------------------")
        print("透視校正完成！")
        print("--------------------------------")
        print(
            "校正尺寸：",
            width,
            "x",
            height
        )

        print(
            "顯示尺寸：",
            display_result.shape[1],
            "x",
            display_result.shape[0]
        )

        print("--------------------------------")


    # -----------------------------------------------------
    # Q：離開
    # -----------------------------------------------------

    elif key == ord('q'):

        break


# =========================================================
# 12. 結束
# =========================================================

cv2.destroyAllWindows()