你的問題在這裡：

hist_equalized = histogram_equalization_numpy(V)

這一行得到的 hist_equalized 是均衡化後的圖片，不是 Histogram。

所以後面：

plt.bar(brightness, hist_equalized, width=1)

會出錯，因為 plt.bar() 需要的是：

x：256 個亮度
y：256 個像素數量

但你的 hist_equalized 是一張 H × W 的圖片。

修改方式

你只要在呼叫 function 後，再計算一次 Histogram：

# 使用 NumPy 做 Histogram Equalization
V_equalized = histogram_equalization_numpy(V)

# 計算 Equalization 後的 Histogram
hist_equalized = np.bincount(
    V_equalized.ravel(),
    minlength=256
)

# 亮度 0~255
brightness = np.arange(256)

# 畫圖
plt.subplot(1, 2, 2)
plt.bar(brightness, hist_equalized, width=1)
plt.title("NumPy Histogram Equalization")
plt.xlabel("Brightness")
plt.ylabel("Number of Pixels")
plt.xlim(0, 255)
你現在的變數應該這樣理解

你的 function：

V_equalized = lut[V]

return V_equalized

所以：

V_equalized = histogram_equalization_numpy(V)

得到的是：

V_equalized
↓
均衡化後的「圖片」

不是：

hist_equalized
↓
Histogram

所以要再做：

hist_equalized = np.bincount(
    V_equalized.ravel(),
    minlength=256
)

才會得到：

亮度 0 → 有幾個 pixel
亮度 1 → 有幾個 pixel
...
亮度 255 → 有幾個 pixel
如果你要完整比較原圖 + NumPy

可以直接這樣寫：

# NumPy Histogram Equalization
V_equalized = histogram_equalization_numpy(V)

# 原始 Histogram
hist_original = np.bincount(
    V.ravel(),
    minlength=256
)

# Equalization 後 Histogram
hist_equalized = np.bincount(
    V_equalized.ravel(),
    minlength=256
)

# 亮度 0~255
brightness = np.arange(256)


plt.figure(figsize=(14, 5))

# 原圖
plt.subplot(1, 2, 1)
plt.bar(brightness, hist_original, width=1)
plt.title("Original Image")
plt.xlabel("Brightness")
plt.ylabel("Number of Pixels")
plt.xlim(0, 255)


# NumPy Equalization
plt.subplot(1, 2, 2)
plt.bar(brightness, hist_equalized, width=1)
plt.title("NumPy Histogram Equalization")
plt.xlabel("Brightness")
plt.ylabel("Number of Pixels")
plt.xlim(0, 255)

plt.tight_layout()
plt.show()

這樣就正確了。

簡單記住：

V
↓
histogram_equalization_numpy()
↓
V_equalized        ← 圖片

V_equalized
↓
np.bincount()
↓
hist_equalized     ← Histogram

另外，你這個 function 裡還有一個特殊情況會除以 0：如果圖片所有像素都是同一個亮度。一般照片通常不會遇到，但如果你要把 function 寫得完整，我可以幫你把這個問題也補上。