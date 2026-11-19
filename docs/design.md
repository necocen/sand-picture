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
  mise.toml             道具の版（trunk など）と、少数のタスク（verify・check・web）。just は入れない
  Cargo.toml            crate は一つ：ライブラリ ＋ 実行ファイル sand-picture ＋ example の道具
  Trunk.toml index.html Web 向けのビルド（mikage の例と同じ形）
  .gitignore            target/ dist/ records/（実験の記録）artifacts/ .dots/ 基準の絵
  AGENTS.md CLAUDE.md README.md   短く書き直す
  assets/               scene.bin, air.bin
  src/
    lib.rs
    model.rs            模型の入口：組み立て、初期化、1 更新の流し込み、リセット、裏返し
    model/
      grid.rs           格子の定数（細かい格子。feature で標準格子に替えられる）と、格子で決まる数値設定
      material.rs       物性の定数（唯一の正本）と、奥行きの抵抗の表
      scene.rs          場面：scene.bin・air.bin、床につながる堆積の分類
      buffers.rs        全バッファの確保と、保存・復元する組
      shader.rs         断片 → 定数 → ループの書き下し（プログラムごとの WGSL）
      shader/unroll.rs
      kernel.rs         プログラムごとの module、入口ごとの pipeline、Kernel
      update.rs         1 更新の手順の表（Step の列）
      slab.rs bed.rs mechanics.rs stream.rs transport.rs labels.rs interface.rs
                        持ち主ごとに、自分の kernel を作って Step を並べる
      shared/ slab/ bed/ mechanics/ stream/ transport/ interface/ …  *.wgsl
    view.rs             三つの描画の切り替え
    view/               simple.rs photoreal.rs flow.rs と *.wgsl
    bin/sand_picture.rs
    bin/sand_picture/   app.rs（読み込み中／実行中）  product.rs（tick、裏返し・リセット、速さ）  panel.rs（操作パネル）
  tools/                example：check（回帰）、run（走行と保存）、bench、replay、lives
  tests/baseline/       回帰の基準：ハッシュの一覧だけを repo に置き、絵は ignore
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

- egui のパネルを一つ（いまの下のメニューと左上の HUD を一つにまとめる）。既定は上の中央。
- 中身：裏返す、再生／停止、リセット、ビュー（単純・リアル・流れ）、時刻、速さ。細かい数値（開始からの平均など）は
  同じパネルの中の折りたたみに入れる。
- キー：F 裏返す、Space 停止、R リセット、V ビュー、H パネルを隠す／出す、P パネルの上下を切り替える。
- 格子の切り替え（G）は無くす。

### 4.2 1 フレームと、描画の頻度

```
絵（単純・リアル・流れ）は、窓の物理の画素数のテクスチャに描いて持っておく
  描き直すのは「新しい更新が完了していて、前に描いてから 1/30 秒たったとき」だけ
窓の再描画（入力、パネル、大きさの変更）は、持っている絵を貼ってパネルを重ねるだけ
```

- 絵は 1 秒に 30 枚まで。いまは更新のたびに 1 枚（0.36 倍速で 1 秒に 45 枚）描いている。砂の動きは画面の上で
  1 秒に数十画素なので、30 枚で足りる。更新が 1 秒に 30 回より遅いときは、更新ごとに 1 枚になる。
- パネルの操作は絵の頻度と関係なく、入力のたびにすぐ再描画する（絵は描き直さないので軽い）。
- mikage の設定は、固定刻みの tick、VSync の提示、要るときだけの再描画。30 という数は定数一つで変えられる。

### 4.3 読み込み中の画面

- 窓をすぐ出し、kernel を 1 フレームあたり十数 ms ぶんずつ作る。そのあいだ、名前と進み具合だけの画面を描く。
- Web では、最初のフレームが出るまで `index.html` の「Loading…」を出しておく（mikage の例と同じ）。
- 起動の時間を測り、十分短ければ進み具合の表示は省く（測った値は「段取り 1 の結果」。窓と Web での値は段取り 2 で見る）。

### 4.4 Web

- `trunk build` で同じ実行ファイルを WASM にする。操作口（agent）と道具はネイティブだけ。
- 時刻は `web-time`、場面は `include_bytes!`。WebGPU の既定の上限（storage buffer 8 本、workgroup 256 スレッド）の
  中で動く作りはいまと同じ。
- 段取りの 0 で空のアプリが Chrome で動くこと、2 で模型が動くことを確かめる。速さもそこで測る。

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
- **WGSL の検査**：`cargo test` が、四つのプログラムを、ループのままの形と書き下した形の両方で naga に通す（GPU は
  要らない）。ループの書き下しは Apple のネイティブ向けだけで入る（`shader.rs` の `WRITE_OUT_LOOPS`）。

段取り 2 に持ち越したもの：読み込み中の画面（抵抗の表の 1.7 秒と、初回の 5.7 秒をどう見せるか）、Web で模型を動かすこと、
この repo 自身の回帰の基準。

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
