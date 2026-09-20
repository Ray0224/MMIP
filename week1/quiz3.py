import cv2
import numpy as np


# ==================================================
# 1. 讀取圖片
# ==================================================

image_path = "data/IMG_3.jpg"

original_image = cv2.imread(image_path)

if original_image is None:
    print("圖片讀取失敗！")
    print("請確認圖片路徑：", image_path)
    exit()

image = original_image.copy()


# ==================================================
# 2. 設定顯示圖片大小
# ==================================================

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
    image,
    (display_width, display_height)
)

show_image = display_image.copy()


# ==================================================
# 3. 儲存使用者點選的座標
# ==================================================

points = []


# ==================================================
# 4. 滑鼠事件
# ==================================================

def mouse_callback(event, x, y, flags, param):

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
                f"第 {len(points)} 個點："
                f"({original_x}, {original_y})"
            )

            # 在顯示圖片上畫點
            cv2.circle(
                show_image,
                (x, y),
                8,
                (0, 0, 255),
                -1
            )

            # 顯示編號
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
                "Select 4 Points",
                show_image
            )


# ==================================================
# 5. 自動排序四個點
# ==================================================

def order_points(points):

    points = np.array(points, dtype=np.float32)

    # 建立新的陣列
    ordered = np.zeros((4, 2), dtype=np.float32)

    # ------------------------------------------
    # 左上：x + y 最小
    # 右下：x + y 最大
    # ------------------------------------------

    sum_points = points.sum(axis=1)

    ordered[0] = points[np.argmin(sum_points)]
    ordered[2] = points[np.argmax(sum_points)]

    # ------------------------------------------
    # 右上：x - y 最大
    # 左下：x - y 最小
    # ------------------------------------------

    diff_points = points[:, 0] - points[:, 1]

    ordered[1] = points[np.argmax(diff_points)]
    ordered[3] = points[np.argmin(diff_points)]

    return ordered


# ==================================================
# 6. 建立視窗
# ==================================================

cv2.namedWindow(
    "Select 4 Points",
    cv2.WINDOW_AUTOSIZE
)

cv2.setMouseCallback(
    "Select 4 Points",
    mouse_callback
)


# ==================================================
# 7. 顯示圖片
# ==================================================

cv2.imshow(
    "Select 4 Points",
    show_image
)

print("--------------------------------")
print("請點選卡片的四個角")
print("--------------------------------")
print("四個點可以任意順序")
print("不用按照左上 → 右上 → 右下 → 左下")
print("--------------------------------")
print("按 Q 可以取消")
print("--------------------------------")


# ==================================================
# 8. 等待使用者點擊
# ==================================================

while True:

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if len(points) == 4:
        break


cv2.destroyAllWindows()


# ==================================================
# 9. 如果成功取得 4 個點
# ==================================================

if len(points) == 4:

    print("--------------------------------")
    print("使用者點選的座標：")
    print(points)
    print("--------------------------------")


    # ==================================================
    # 10. 自動排序
    # ==================================================

    ordered_points = order_points(points)

    print("自動排序後：")
    print("左上：", ordered_points[0])
    print("右上：", ordered_points[1])
    print("右下：", ordered_points[2])
    print("左下：", ordered_points[3])
    print("--------------------------------")


    # ==================================================
    # 11. 計算校正後寬度
    # ==================================================

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


    # ==================================================
    # 12. 計算校正後高度
    # ==================================================

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


    print("校正尺寸：")
    print("Width :", width)
    print("Height:", height)
    print("--------------------------------")


    # ==================================================
    # 13. 建立 Perspective Transformation
    # ==================================================

    src = ordered_points

    dst = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])


    # ==================================================
    # 14. 計算轉換矩陣
    # ==================================================

    M = cv2.getPerspectiveTransform(
        src,
        dst
    )


    # ==================================================
    # 15. 執行透視轉換
    # ==================================================

    result = cv2.warpPerspective(
        original_image,
        M,
        (width, height)
    )


    # ==================================================
    # 16. 限制顯示尺寸 1280 × 720
    # ==================================================

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


    # ==================================================
    # 17. 顯示結果
    # ==================================================

    cv2.imshow(
        "Perspective Result",
        display_result
    )

    print("--------------------------------")
    print("校正完成！")
    print(
        "顯示尺寸：",
        display_result.shape[1],
        "x",
        display_result.shape[0]
    )
    print("--------------------------------")
    print("按任意鍵結束")


    cv2.waitKey(0)
    cv2.destroyAllWindows()


else:

    print("--------------------------------")
    print("沒有選擇完整的 4 個點")
    print("--------------------------------")