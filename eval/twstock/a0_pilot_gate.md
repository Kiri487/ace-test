# A0 pilot 關卡判準（事前寫死）

狀態：**草稿，待使用者確認**。確認後 commit，commit 之後不得修改；
A0 pilot 的 manifest 須記錄本檔的 git commit 與 sha256，執行時若本檔與該 commit 不一致即拒絕執行。

撰寫日：2026-09-15。撰寫時尚未有任何 A0 評分（先前 5 次格式試跑只看輸出格式，未計算 IC）。

## 1. 這一關回答什麼

A0 pilot（`arm_protocol.PILOT_A0`：A0、1 個 seed、後截止日 85 個決策點）只回答一件事：
**只給新聞標題時，模型的評分是否帶有排序資訊，使 A1／A2 的比較有意義**（v9.3 §9）。

pilot 不是論文的 A0（未與 A1、A2 同日交錯，違反 §5.2.0），數字不進任何結果表。

## 2. 指標：csIC，h=10

```json
{
  "metric": "mean over decision dates of daily csIC at h=10",
  "daily_csic": "ic.cs_ic: Spearman(score, alpha_h10) over universe members with both values; NaN when fewer than ic.CS_MIN_PAIRS (20) pairs",
  "dates": "the 85 post-cutoff decision dates 2026-04-27..2026-08-26 (replay.CONDITIONS['post']); void dates and NaN csIC dates excluded",
  "horizon": 10
}
```

**為何是 csIC、不是 tsIC：**

1. **tsIC 有機械偏誤，且對 LLM 評分的方向未知。** §6.1 的安慰劑基準線（85 天窗口、h=10 為 −0.189）是**動能評分**
   的值：偏誤來自評分與該股自身過去報酬的關係。LLM 評分與過去報酬的關係在執行前不知道，偏誤可正可負
   （§6.1 第 3 點：須逐模型量測），因此事前無法寫出一條適用於 A0 的 tsIC 基準線。拿 −0.189 或 0 當基準都不對。
2. **csIC 不受這個偏誤影響，也不受基準選擇影響**（§4.2：同日對所有股票減同一個數，不改排序）。
3. **csIC 有已凍結的虛無分布**：隨機評分的逐日 csIC（50 seeds × 399 日，h=10）平均 0.0000、標準差 0.144
   （`results/feedback_null/pooled_random_cs_h10.npy`）。基準線就是 0，且有來源。
4. **h=10** 是驅動 A1／A2 回饋與組合持有期的視野（§4.1、§5.1）。

tsIC（四個視野）照算、照報告，**不參與判定**。

## 3. 門檻

```json
{
  "pass": "mean_csic_h10 > 0.0",
  "fail": "mean_csic_h10 <= 0.0",
  "invalid_not_judged": [
    "more than 9 of the 85 dates void (over 10%)",
    "fewer than 76 dates with a computable csIC at h=10"
  ]
}
```

判定「不通過」時不自動觸發任何退路；依 §9 的兩個退路（（a）當期 LLM 摘要、（b）純數值輸入）由使用者決定。
「無效」時先查原因（作廢種類、NaN 來源），不判定、不重跑到有結果為止。

## 4. 這是描述性門檻，不是統計檢定

- 85 個決策點、h=10 重疊取樣，N_eff ≈ 8.5。
- 以凍結的虛無 sd 0.144 計，85 日 csIC 平均值的標準誤約 0.144 / √8.5 ≈ **0.049**；
  若以實測的逐日離散度計會更大。這與欲偵測的效果量（文獻中 IC 0.05 已屬不錯）同一量級。
- 因此**不產生 p 值、不報 t 值**（§6.1：N_eff < 20 不報 t）。

門檻 0 的判定性質（常態近似，SE 0.049，僅供理解，非檢定力分析）：

| 真實平均 csIC | 0 | +0.025 | +0.05 | +0.10 |
|---|---|---|---|---|
| P(通過) | 0.50 | 0.69 | 0.84 | 0.98 |

也就是：**完全沒有訊號的模型有一半機率通過**。這一關只能排除「排序方向為零或相反」這種明顯情況，
不能證明標題「足夠」。選 0 而非更高門檻的理由：門檻訂在 +0.05 時，真實 IC 0.05 的模型也只有一半機率通過；
在階段一「只估效果量、不做推論」（§5.0）的定位下，錯殺一個有訊號的輸入（須改輸入並重做離線驗證）
比放行一個無訊號的輸入代價更高。

## 5. 一併報告、不參與判定

- csIC 平均：h=5／20／40；Cold-Start（前 1/3）與 Exploitation（後 2/3）分段
- tsIC 平均：h=5／10／20／40，註明「LLM 評分的安慰劑基準線未定義，不可與 0 或 −0.189 比較」
- 作廢數與作廢種類、重試數、final_answer 形式分布
- 每日 csIC 的實測標準差（與凍結虛無 sd 0.144 並列）
- 實際呼叫數、token、成本、延遲，與預算的 pilot 估計並列
- 資料快照 id、虛無分布的來源版本（`data_snapshot.FEEDBACK_NULL`）
