# 作り直しの設計（sand-picture2）

2026-11-19 にユーザーが承認した計画である。過去の記録との照合と、使えるデータの一覧は、移植の元の `../sand-picture/docs/rewrite/records.md`。

## Context

いまの repo は履歴が重く（`.git` 32 GiB）、コードには経緯の跡が残っている。v2.20 と同じ計算をするものを、
隣の `sand-picture2` にゼロから書き直す。模型・式・段階の順序・反復の回数は変えない（泡を輪郭で持つ案と、解法を
収束させる案は、記録で既に退けられていた：`docs/rewrite/records.md`）。変えるのはプログラムの組み立て方と
アプリである。

ユーザーの指定（2026-11-19）：

- crate は分けない。モジュールの境界を仕掛けで守ることもしない。
- 格子は細かい方（270×180）だけ。実行時の切り替えは無し。ビルドの切り替えはあってもよい。
- 実験の記録は最初から ignore して、repo に一切入れない。
- フォルダは `/Users/necocen/Development/sand-picture2`。GitHub に置くときのことは、そのとき考える。
- trunk で Web 向けにもビルドできること。ツールチェーンは mise と `rust-toolchain.toml`。just は要れば。
  整備の手本は nocturne12。
- 操作パネルは egui で一つだけ。隠せる。既定は上（下は模様を見たい）。上下を切り替えられると理想。
- ビューは今と同じ三つ（単純・リアル・流れ）。リアルの美麗さは、あとからしっかり調整する。
- 初回の読み込みが長いならスプラッシュ。
- 描画のフレームレートは、低すぎず、計算を圧迫しない程度に。
- mikage は軽い描画の枠組みのまま使う。窓は論理画素。環境で効き方の変わる最適化は増やさない。
  （計画の時点で mikage に頼んでいた #30 は、前提にした読みが誤りだった。下の 2.3）

**承認で始めたこと**：段取りの 0（土台）と 1（模型の移植と v2.20 との照合）。それぞれの終わりで報告する。
2 以降は、1 の結果を見てから進める。旧 repo（`../sand-picture`）は読むだけの参照にして、書き換えない。

---

## 1. 土台とモジュール構造

```
sand-picture2/
  rust-toolchain.toml   Rust の版はここで固定（1.95.0＝いまの回帰の環境。rustfmt・clippy、wasm32-unknown-unknown）
  mise.toml             道具の版（trunk など）と、少数のタスク（verify・check・check-full・web）。just は入れない
  Cargo.toml            crate は一つ：ライブラリ ＋ 実行ファイル sand-picture ＋ example の道具
  Trunk.toml index.html Web 向けのビルド（mikage の例と同じ形）
  .gitignore            target/ dist/ records/（実験の記録）artifacts/ .dots/ 基準の絵
  AGENTS.md CLAUDE.md README.md   短く書き直す
  assets/               scene.bin, air.bin
  src/
    lib.rs
    wgsl.rs             WGSL をコンパイルに渡す形にする：ループの書き下し（対象ごとのスイッチ）と、その検査
    wgsl/unroll.rs
    model.rs            模型の入口：少しずつの組み立て（Building）、1 更新の流し込み、リセット、裏返し
    model/
      grid.rs           格子の定数（細かい格子だけ）と、格子で決まる数値設定
      material.rs       物性の定数（唯一の正本）と、奥行きの抵抗の表（帯ごとに計算する）
      scene.rs          場面：scene.bin・air.bin、床につながる堆積の分類
      buffers.rs        全バッファの確保、場面の書き込み、リセットのときの戻し
      shader.rs         断片 → 定数（プログラムごとの WGSL）
      kernel.rs         プログラムごとの module、入口ごとの pipeline、Kernel
      update.rs         1 更新の手順の表（Step の列）
      slab.rs bed.rs mechanics.rs stream.rs transport.rs labels.rs interface.rs
                        持ち主ごとに、自分の kernel を作って Step を並べる
      shared/ slab/ bed/ mechanics/ stream/ transport/ interface/ …  *.wgsl
    view.rs             三つのビュー（Views）と、その切り替え（Mode）
    view/               picture.rs（全面の三角形 1 枚：単純・流れ）  photoreal.rs  と *.wgsl
    bin/sand_picture.rs
    bin/sand_picture/   app.rs（組み立て中／実行中、キー、操作口）  product.rs（更新、裏返し・リセット、速さ）
                        screen.rs（持っている絵と、窓への貼り付け）  panel.rs（操作パネル）
  tools/                example：check（回帰）、run（走行と保存）。common.rs は両方が使う GPU と絵の読み出し
  tests/                README.md（回帰の手引き）と baseline/hashes.txt（ハッシュだけ。絵は repo に入れない）
```

- nocturne12 にならう点：Rust の版は `rust-toolchain.toml`、ほかの道具の版は `mise.toml`、`.gitignore` は用途ごとに
  見出しを付ける。タスクは 3 つほどなので mise のタスクで足り、just は入れない。
- 旧 repo から写すもの：WGSL の kernel の本文、`grid`・`material`・場面と初期の堆積・裏返しの計算。
  書き直すもの：バッファと kernel の管理、シェーダーの組み立て、アプリ、道具。
- 実験の記録（走行、計測、検討のメモ）は `records/` に置き、最初から ignore する。旧 repo の記録も持ち込まない。

## 2. データ構造

### 2.1 手順の表

```rust
pub struct Kernel { entry: &'static str, pipeline: wgpu::ComputePipeline,
                    bindings: wgpu::BindGroup, groups: Groups }   // Groups：固定の数か、間接
pub struct Step   { pub stage: Stage, pub kernel: usize }         // kernel は Kernels の中の位置
pub struct Kernels { kernels: Vec<Kernel> }                       // 1 更新は Vec<Step>（608 個）
```

- module はプログラムごとに 1 つ（模型・流れの解法・輸送の分割の予定・鏡映の 4 つ）、pipeline は入口ごとに 1 つ。
  v2.20 は 128 個の Kernel がそれぞれ module と pipeline を作っていた。
- 1 更新を、モジュールごとの `encode` の手続きではなく `Step` の列として持つ。製品は 1 つの pass に流し、道具は
  同じ列を段階ごと・kernel ごとに分けて測り、任意の Step の前後で書き出し、1 つだけ再生する。
- kernel は少しずつ作れるようにする（読み込み中の画面を出すため。下の 4.3）。

### 2.2 バッファ

寿命で三つ。**状態**と**持ち越し**は保存して回帰で比べ、**作業**は保存も比較もしない。

| バッファ | 中身 | 寿命 |
|---|---|---|
| `cells`（96 B × セル） | 四相の量、運動量、粒子の応力、粉、堆積 | 状態 |
| `interface`（vec4 × 23 行） | 識別子ごとの気体の量と平面、接触形状 | 状態・持ち越し |
| `history`（× 3 行） | 応力の解法のときの開口と状態、板の牽引力 | 状態 |
| `psi`・`films`・`strata` | 流れ関数、泡の膜の計数と前回の成分、堆積の層の記録 | 状態 |
| `faces`（160 B × 面）・`terrain` | 面の応答と開口、堆積の列 | 持ち越し |
| `bed_view`（3 行）・`slab`（5 行）・`mech`（13 行）・`twork`（39 行） | 堆積の見え方、上の砂、力学の作業、輸送の作業 | 作業 |
| `transport`・`members`・`plan`・流れの解法の作業ほか | 流束、グループ、分割の予定、行列 | 作業 |

- いまの `geometry`（64 行、使い回しあり）を `interface`・`history`・`bed_view`・`slab`・`mech`・`twork` の六つに
  分ける。用途ごとに 1 本にはできない（輸送の kernel がいまも 8 本ちょうど）。六つなら、シェーダーの文字列から
  数えた上限で 9 個の kernel がちょうど 8 本。越えるものが出たら `members` を `twork` に入れる。
- これは段取りの 4 で行う。1 では v2.20 と同じ並びのまま移す（照合を先に通すため）。
- 保存する組が足りているかは、「保存 → 復元 → 続き」が「止めずに走った続き」とビットで一致することで確かめる。

### 2.3 シェーダー

`断片 ＋ 定数 → ループの書き下し → プログラムごとの module → 入口ごとの pipeline`

- module はプログラムごとに 1 つだけ作る。v2.20 の起動が重かったのは、同じプログラムの全文を kernel ごとに解析・検証
  し直していたためである（1 回 約 22 ms）。
- 入口に届く宣言だけを集める段は置かない。計画の時点では「wgpu の Metal 側は pipeline ごとに module の全関数を訳して
  コンパイルする」と読んでいたが、誤りだった（2026-11-19）：naga 30.0.1 は、入口が複数ある module から pipeline を
  作るとき、その入口に届くものだけに縮めてから MSL を書く（`process_overrides` の中の `compact`）。入口ごとに刈り込んだ
  module を作る実装も書いて測ったが、プログラムごとに 1 つの module と起動の時間が同じ（1.9 秒）だったので外した。
- 書き下しは最後の、スイッチで外せる段。ネイティブ（Metal）では v2.20 と同じ書き下しを使い、Web では測るまで
  外しておく。

## 3. 呼び出し順序

### 3.1 一生

```
起動     窓を出す → 読み込み中（kernel を数フレームに分けて作り、進み具合を描く）→ 初期化の pass → 実行中
毎 tick  下の 3.2
毎フレーム 下の 4.2
リセット  状態のバッファを書き直し、初期化の pass を流す（pipeline は作り直さない）
```

初期化の pass：`bed_geometry → bed_release → 界面（平面・接触・開口）→ prepare_bed_occupancy →
prepare_bed_passage → predict → prepare_grain_stress → prepare_stream → cache_contact → interface_tensor →
assemble → scale_interpolation・coarsen`

### 3.2 1 tick（8 ms）

```
1. 保留中のリセット・裏返しを適用
2. 裏返しを 1 更新ぶん進める（この更新の重力と横向きの加速度を書く。水平を過ぎたら鏡映の合図）
3. compute pass を 1 つ：鏡映があれば invert_cells → invert_stream → 界面の作り直し → clear_strata、
   そのあと 608 Step
4. 完了の通知で、進んだ更新と速さを数える
```

### 3.3 1 更新（608 dispatch）

| 段階 | kernel（順） | 数 |
|---|---|---:|
| Slab | teeth_underside, body_faces, gap_marks, column_cells | 4 |
| BedPrepare | begin_displacement, bed_geometry, record_strata, bed_release, prepare_bed_occupancy, prepare_bed_passage | 6 |
| FirstPredict | predict, transfer_contact, prepare_grain_stress, prepare_stream | 4 |
| FirstTensor・Assemble・Coarsen | interface_tensor ／ assemble ／ scale_interpolation, coarsen × 5 | 8 |
| FirstSolve | anchor_plan, fcg_start, fcg_residual,（V-cycle 66 ＋ 共役勾配法 5）× 2, fcg_finish | 146 |
| FirstReturn | prepare_grain_velocity, grain_constraint,（pressure_red, pressure_black）× 34, 最後の組, copy_stress, project_faces, capture_bed_source | 75 |
| Capture・Surface・BedRefresh | begin_displacement, bed_geometry, bed_exchange ／ surface_even, surface_odd ／ prepare_bed_occupancy, prepare_bed_passage | 7 |
| SecondContact | predict, prepare_grain_stress, transfer_contact, prepare_stream, prepare_contact_velocity, contact_constraint, copy_stress, prepare_grain_stress, prepare_stream | 9 |
| SecondTensor・Assemble・Coarsen | 一回目と同じ（assemble は初期値を捨てる側） | 8 |
| SecondSolve | 一回目と同じ形で 3 反復 | 217 |
| SecondReturn | project_faces, apply_momentum, hold_friction | 3 |
| TransportPrepare | free_powder, plug_record, request・choose・resolve・accept_groups, 分割の予定（reset, reduce, finish） | 9 |
| TransportCarry | 12 枠 ×（reconstruct_common, layer_flux, gas_capacity, compose_layer_flux, material_capacity, limit_layer_flux, advect_groups, advance） | 96 |
| TraceGas | clear_trace_gas, plug_descent | 2 |
| Labels | label_init, merge, roots, sizes, records, measure, color, apply | 8 |
| Interface | reconstruct_interface ／ reset_contact_work, plan_contacts, seal_contact_work, fit_contacts（間接）／ wet_apertures | 6 |

V-cycle の 66：`anchor_predict, anchor_apply`、階層 3 つを下りながら `relax × 9, evaluate_residual, restrict_rhs`、
`coarse_cycle`、上りながら `prolong, relax × 9`。

## 4. アプリ

### 4.1 操作パネルとビュー

- egui のパネルを一つ（v2.20 の下のメニューと左上の HUD を一つにまとめた）。既定は上の中央。
- 中身：裏返す、再生／停止、リセット、ビュー（単純・リアル・流れ）、時刻、速さ。細かい数値（開始からの平均、
  更新の数、水槽と格子、キーの一覧）は同じパネルの「Details」で開く。
- キー：F 裏返す、Space 停止、R リセット、V ビュー、H パネルを隠す／出す、P パネルの上下を切り替える。
- 格子の切り替え（G）は無くす。

### 4.2 1 フレームと、描画の頻度

```
絵（単純・リアル・流れ）は、テクスチャに描いて持っておく（段取り 2 では 1440×960。窓の画素数にするのは段取り 3）
  描き直すのは「前に描いたあとに更新かビューの変更があって、前に描いてから 1/30 秒たったとき」だけ
窓の再描画（入力、パネル、大きさの変更）は、持っている絵を貼ってパネルを重ねるだけ
```

- 絵は 1 秒に 30 枚まで。いまは更新のたびに 1 枚（0.36 倍速で 1 秒に 45 枚）描いている。砂の動きは画面の上で
  1 秒に数十画素なので、30 枚で足りる。更新が 1 秒に 30 回より遅いときは、更新ごとに 1 枚になる。
- パネルの操作は絵の頻度と関係なく、入力のたびにすぐ再描画する（絵は描き直さないので軽い）。
- mikage の設定は、固定刻みの tick（8 ms）と、要るときだけの再描画。提示と、同時に流す提出の数は mikage の既定のまま
  （描けたフレームから提示する Immediate、提出は 8 つまで）。計画では「VSync の提示」（フレームを 1 つずつ流す）と
  していたが、測って替えた（「段取り 2 の結果」の窓の速さ）。30 という数は定数一つ（`PICTURE_INTERVAL`）で変えられる。
- 絵は、描いた枚数の履歴によらない（更新のたびに描いても、確かめる時刻にだけ描いても、同じ画素）。

### 4.3 読み込み中の画面

- 窓をすぐ出し、模型を 1 フレームあたり 30 ms ぶんずつ組み立てる（抵抗の表の帯、バッファ、kernel を 1 つずつ）。
  そのあいだ、何をしているかと進み具合のバーだけの画面を描く。模型ができたらビューを作り、更新を始める。
- kernel が全部できたところで 1 フレーム描いてから、最初の仕事を GPU に渡す。ブラウザは pipeline をそのときに
  コンパイルするので（wgpu の Web 側は作る呼び出しがすぐ返る）、その数秒のあいだ「kernel をコンパイル中」が出ている。
- Web では、最初のフレームが出るまで `index.html` の「Loading…」を出しておく（mikage の例と同じ）。

### 4.4 Web

- `trunk build` で同じ実行ファイルを WASM にする。操作口（agent）と道具はネイティブだけ。
- 時刻は `web-time`、場面は `include_bytes!`。WebGPU の既定の上限（storage buffer 8 本、workgroup 256 スレッド）の
  中で動く作りはいまと同じ。
- 段取りの 0 で空のアプリが Chrome で動くこと、2 で模型が動くことを確かめた（結果は「段取り 2 の結果」）。
- ループの書き下しは Web では入れない（測って、得が無かった）。

## 5. 検証

1. **v2.20 との照合**：v2.20 のバイナリが 30 秒ごとに保存した状態（旧 repo の `artifacts/local/limits` の
   `loops2-life`・`adopt-life2`）と、状態をビット単位で比べる。ずれた更新は Step ごとに再生して、最初に違う
   kernel を特定する。写す途中の確認には、feature で標準格子にして `adopt-std` と比べる（2 倍速い）。
2. **回帰**：状態と、単純・リアルの画素のビット一致（0・1・5 秒と、60 秒まで）。基準は 1 が通った時点で取る。
3. **保存と復元**：保存 → 復元 → 続きが、止めずに走った続きと一致すること。
4. **見た目の変わる変更**（窓の画素で描く、粒の見え方、あとの美麗さの調整）：同じ時刻の前後の絵でユーザーが判断する。
5. **丸めの変わる変更**：いまと同じ手順（丸め違いの起動 10 本ずつ、一生の読み取り）。読み取りは `tools` の `lives`。

GPU を使う照合は、ユーザーがアプリを見ていないときに流す。流すときは知らせる。

## 6. 段取り

0. **土台**：フォルダ、ツールチェーン、ignore、空のアプリ（窓・パネル・読み込み中の画面）がネイティブと
   `trunk build` の両方で動くところまで。→ 報告
1. **模型の移植**：バッファは v2.20 の並びのまま、kernel を写し、手順の表で組む。
   検証の 1 が通るまで。起動の時間を測る。→ 報告
2. **アプリ**：tick、裏返し、作り直さないリセット、三つのビュー、描画の頻度、Web で動かす。回帰の基準を取る。
3. **窓の画素で描く**：絵の大きさを窓に合わせ、粒の見え方を画素の大きさから決める。1440×960 では基準と一致。
   ほかの大きさの絵を見せる。
4. **バッファの分割**：まとまりごとに一つずつ、そのたびに照合。
5. **リアルの調整**：落ちる砂を粒で描く、上の砂の縁をくっきり描く、ほか。一つずつ、前後の絵で。

## 段取り 1 の結果（2026-11-19）

模型は `src/model` にある。バッファは v2.20 の並びのまま、1 更新は 24 段階・608 step の表で、段階ごとの数は 3.3 の表と
同じである（`cargo run --release --example run -- OUT --steps 0` が表を出す）。

**v2.20 との照合**（M1 Max、macOS 27.2.1、wgpu・naga 30.0.1、Rust 1.95.0。照合の道具と走行は `records/stage1/`）

| 何を | 比べた先（旧 repo） | 結果 |
|---|---|---|
| 0・1・5・15・30・60 秒の、比べる 8 本のバッファ | 回帰の基準 `tests/baseline/manifest.json` のハッシュ | 48 個すべて一致 |
| 起動からの一生（1200 秒、15 万更新）：30 秒ごとのセルと堆積の列 | v2.20 の走行 `loops2-life` の 40 回の保存状態 | 80 ファイルすべて一致 |
| 同じ一生の 1200 秒の時点：8 本と strata の全体 | `loops2-life/state-01200` | 9 本すべて一致 |
| その終わりを裏返した二回目の一生（v2.20 の 1200 秒の状態を読み込み、2400 秒まで）：30 秒ごとのセルと堆積の列 | v2.20 の走行 `adopt-life2` の 40 回の保存状態 | 80 ファイルすべて一致 |
| 二回目の一生の終わり（2400 秒）：8 本と strata の全体 | `adopt-life2/state-02400` | 9 本すべて一致 |
| 保存 → 読み込み → 続き（15 秒の保存から 30 秒まで） | 止めずに走った 30 秒の状態 | 8 本と strata が一致 |

- **比べられないもの**：`labels` のうち、泡の記録の番号が入る部分（根ごとの欄、記録の並び、セルごとの「留められた
  識別子の記録」の番号）。番号は GPU のスレッドが記録に着いた順（atomic の加算）で決まり、同じビルドを 2 回走らせても
  違う。番号と記録は同じ付け直しの中でだけ対で使われるので、ほかのバッファには違いが出ない。同じバッファの中の
  節点、膜の計数（押されている・留められている）、セルの語は一致する。v2.20 も同じである。段取り 4 でバッファを
  分けるときは、番号の入る部分と、更新をまたいで持つ計数・語を別にする。
- **起動の時間**（画面なしの道具、ほかの走行が無い静かな機械）：プロセスの開始から模型が使えるまで 1.9 秒（Metal に
  pipeline のキャッシュがあるとき）、5.7 秒（Metal が 91 本の pipeline をコンパイルするとき：初回と、シェーダーを変えた
  あと）。どちらも、うち 1.7 秒は CPU での抵抗の表（`material::table`）の計算である。同じ条件・同じ時間帯に測った v2.20 の
  道具は、8 更新を含むプロセス全体で 4.5 秒、こちらは 2.1 秒。計画に書いていた「14 秒」は、アプリが動いている温まった
  機械での値で、同じ条件の比較ではなかった。
- **速さ**：v2.20 の道具（`trace`）と同じ。同じ 60 秒の保存状態から同じ 1250 更新を、画面なしで交互に 3 回ずつ
  （熱の圧力 2 の温まった機械、GPU の走行はほかに無し）：v2.20 が 1 更新 23.4・19.7・21.0 ms、こちらが 21.1・22.1・20.8 ms
  （平均 21.4 と 21.3）。回ごとの揺れ（±8%）の中で差は無い。dispatch の列もシェーダーも同じなので、そうなるはずの結果である。
- **WGSL の検査**：`cargo test` が、四つのプログラムを、ループのままの形と書き下した形の両方で naga に通す（GPU は
  要らない）。ループの書き下しは Apple のネイティブ向けだけで入る（`shader.rs` の `WRITE_OUT_LOOPS`）。

段取り 2 に持ち越したもの：読み込み中の画面（抵抗の表の 1.7 秒と、初回の 5.7 秒をどう見せるか）、Web で模型を動かすこと、
この repo 自身の回帰の基準。

## 段取り 2 の結果（2026-11-19）

窓が模型を動かす。製品の場面が静止した状態から、起動と同時に裏返る。ビューは三つ、パネルは一つで上にあり、隠せて、
下へ移せる。絵は v2.20 と同じである（模型も、ビューのシェーダーも変えていない）。

**確かめたこと**（M1 Max、macOS 27.2.1。道具と記録は `records/stage2/`）

| 何を | どう確かめたか | 結果 |
|---|---|---|
| 三つのビューの移植 | Photoreal と Simple の画素を、v2.20 の基準の 0・1・5・15・30・60 秒と比べた（8 本のバッファも同時に） | 60 個のハッシュがすべて一致 |
| 絵が、描いた枚数の履歴によらないこと | 更新のたびに Photoreal を描いた走行と、確かめる時刻にだけ描いた走行の、三つのビューの画素 | 0・1・5 秒で同じ |
| 作り直さないリセット | 5 秒でリセットし、1 秒進めた状態を、v2.20 の基準の 1 秒と比べた。`check` は毎回、リセット後の 0・1・5 秒が最初と同じことを確かめる | 8 本と絵がすべて一致 |
| 窓そのものの絵 | 操作口で時間を止めてリセットし、0・125・625 更新まで進めて、窓が持っている絵を読んだ | v2.20 の基準の絵と画素単位で一致 |
| 窓の起動 | プロセスの開始から最初の更新が走るまで（Metal に pipeline のキャッシュがあるとき、3 回） | 2.4〜2.7 秒。そのあいだ進み具合の画面を 63 枚描く |
| Web 版 | ヘッドレスの Chrome（WebGPU、Metal）で動かし、パネルの時計を 60 秒はさんで読んだ | 同じ絵が動く。0.25 倍速。起動は初回 約 9 秒、2 回目から 約 4 秒。wasm は 4.57 MB（gzip 2.04 MB） |
| Web でのループの書き下し | 書き下した Web 版を同じように 60 秒 | 0.27 倍速（回ごとの揺れの中）、初回の起動は 24 秒。Web には入れない |

- **窓の速さと、mikage の設定**（絵の時刻 5→20 秒、Photoreal、熱の圧力 1 まで冷ました機械から順に走らせ、GPU の電力と
  周波数を毎秒読んだ。同じ冷ましの回の中でだけ比べられる。`records/stage2/residency*.txt`）：

  | 設定 | 速さ（倍速） | 提示したフレーム/秒 | GPU の電力・周波数 |
  |---|---|---|---|
  | 回 A：v2.20 の窓（Immediate、提出 8、毎フレーム再描画） | 0.352・0.357 | 19・20 | 9.5 W、1001・1011 MHz |
  | 回 A：**採用した設定の製品ビルド**（Immediate、提出 8、要るときだけ再描画） | 0.356・0.360 | 20 | 9.0・9.5 W、995・1010 MHz |
  | 回 B：v2.20 の窓 | 0.377・0.379 | 21 | 11.0・11.1 W、1080・1084 MHz |
  | 回 B：同じ設定の、設定を外から替えられる試験用ビルド | 0.402・0.414 | 23 | 12.1・13.0 W、1156・1193 MHz |
  | 回 B：VSync 提示、提出 3 | 0.389 | 16 | 11.2 W、1068 MHz |
  | 回 B：Immediate、提出 3 | 0.344 | 72 | 10.5 W、1074 MHz |
  | 回 C：v2.20 の窓 | 0.346・0.354 | 19 | 8.8・9.9 W、975・1014 MHz |
  | 回 C：VSync 提示、提出 2（最初に作った設定） | 0.326・0.328 | 20・21 | 8.6・8.5 W、940・957 MHz |

  - 先に流しておく更新が少ないと、GPU は止まってはいない（停止の割合は 0%）のに、低い周波数と電力で動く。更新と更新の
    あいだにキューが空になる短い隙間があり、周波数を決める仕組みがそれを「忙しくない」と見るためと読める（周波数と
    電力の測定までが事実で、仕組みは推測）。最初に作った設定（VSync 提示、提出 2）は v2.20 の窓より 7% 遅かった。
    採用した設定は v2.20 の窓と同じ速さである（製品ビルドで +1%。試験用ビルドの回では +6〜9% だったが、回ごとの
    機械の状態の違いが同じ程度あるので、「速くなった」とは言わない）。
  - VSync 提示は、流せるフレームが 1 つなので、深いキューの後ろでフレームが待たされる。熱の圧力 2 の温まった機械で
    提示できたフレームは、提出 2 で毎秒 17〜18、3 で 12、4 で 8、8 で 4〜5（v2.20 の窓は 15）。
  - 熱の圧力 2 まで温まった機械では、どの設定も同じ速さになる（0.26〜0.29 倍速。GPU に許される電力で決まるため）。
- **この repo の回帰**：`check`（`mise run check`、60 秒までは `check-full`）と `tests/baseline/hashes.txt`。確かめる点は
  7 つ（0・1・5 秒、二度目の裏返しの 1 秒後、15・30・60 秒）、それぞれ 12 項目（8 本、strata、三つのビューの画素）。
  最初の基準は v2.20 そのもので、v2.20 の基準にもある 60 項目はすべて同じ値である。手引きは `tests/README.md`。
- **ブラウザでの進み具合の表示**：wgpu の Web 側は pipeline を作る呼び出しがすぐ返り、ブラウザは最初の仕事が届いたときに
  コンパイルする。kernel を「1 つずつ作る」進み具合は一瞬で終わり、そのあと数秒止まる。その前に 1 枚描くようにした。
- **操作口**：mikage は、自動の tick のあとには再描画を頼むが、操作口から進めた tick のあとには頼まない。アプリの側で、
  更新のたびに再描画を頼む。

段取り 3 以降に持ち越したもの：絵の大きさを窓の画素数に合わせること（いまは 1440×960 の絵を窓に合わせて貼っている）、
`geometry` の分割、リアルの調整。

## 決めていないこと（既定の案）

- GitHub に置くときの名前、公開するか、ライセンス：そのときに。
- 物理の仕様（report.md、docs/physics）：当面は旧 repo に置いたまま参照する。
- Rust の版：照合が通るまで 1.95.0。上げるのはそのあと、基準を取り直してから。
- `.dots/`：ignore して手元だけで使う（履歴を軽く保つため）。

## 参照するファイル

- 写す元（旧 repo、読むだけ）：`src/picture_flow/**/*.wgsl`、`grid.rs`・`material.rs`・`initial.rs`・
  `bed_initial.rs`・`mod.rs`（裏返し）・`unroll.rs`、`src/bin/sand_picture/*.rs`、`tools/trace.rs`・`regression.rs`。
- 順序と寿命の正本：旧 repo の `docs/physics/state-update.md`（S1・S2）、`src/picture_flow/stage.rs`。
- 整備の手本：nocturne12 の `rust-toolchain.toml`・`mise.toml`・`.gitignore`、mikage の `Trunk.toml`・`index.html`。
- 設計の記録：旧 repo の `docs/rewrite/design.md` をこの内容に直す。新しい repo には `docs/design.md` として置く。

## 確かめ方（段取り 0・1 の終わり）

- 0：`cargo run --release` で窓・パネル・読み込み中の画面が出る。`trunk build` が通り、Chrome で同じものが出る。
  `cargo fmt --check`・`cargo clippy --all-targets`・`cargo test` が通る。
- 1：`cargo run --release --example run -- …` の保存状態が、v2.20 の 30 秒ごとの状態と全部一致する（起動からの
  一生と、裏返した二回目）。起動から最初の更新までの秒数を、v2.20 の道具と同じ条件で並べて報告する。
