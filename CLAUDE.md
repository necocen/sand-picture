# Claude 向けガイド

製品の要件・コード規約・進め方の正本は AGENTS.md。

@AGENTS.md

## 現在の段階（2026-11-19〜）

- 設計と段取りは [docs/design.md](docs/design.md)（2026-11-19 にユーザーが承認）。
- 段取りの 0（土台：ツールチェーン、ignore、窓と操作パネルだけのアプリ、Web 向けのビルド）は済み。
- 段取りの 1（模型の移植と、v2.20 との照合）は済み。模型は `src/model` にあり、v2.20 とビット単位で同じ結果を出す
  （docs/design.md「段取り 1 の結果」）。アプリにはまだ載っていない：窓に出るのはパネルだけである。
- 2 以降（アプリに模型を載せる、窓の画素で描く、バッファの分割、リアルの調整）は、ユーザーが進めると言ってから始める。

## 決まりごと

- コミットはこまめに、contextual commit で（`.claude/skills/contextual-commit`）。確かめの付いた一区切りごとに 1 つ。
  あとの変更を前のコミットへ amend で畳み込まない（2026-11-19 ユーザー）。
- wgpu と naga は 30.0.1 に固定してある（v2.20 と同じ。naga が WGSL を訳す形が変わると結果のビットが変わりうる）。
  Rust も 1.95.0 に固定。どちらも、v2.20 との照合が通ってから、回帰の基準を取り直して上げる。
- GPU を使う実行（アプリ、照合、計測）は sandbox の外で行う。
- 模型（`src/model`）に手を入れたら、結果が変わらないことを v2.20 との照合で確かめる：`run` の保存状態を、旧 repo の
  回帰の基準（`../sand-picture/tests/baseline/manifest.json` の 0〜60 秒のハッシュ）と比べる。比べる手順と道具は
  `records/stage1/`（ignore 済み、この機械だけ）にある。この repo 自身の基準は段取り 2 で取る。
- `labels` バッファは走行どうしで一致しない（泡の記録の番号が GPU のスレッドの順で決まる）。比べるのは 8 本と strata。
- 速さや起動の時間を比べるときは、同じ日に同じ条件で両方を測る（機械の状態で 3 倍変わる）。
- ユーザーがアプリを見ているあいだは、GPU や CPU を長く使う走行を自分から始めない。流すときは知らせる。
- `../sand-picture` の外と、この repo の外には書かない。`../sand-picture` は読むだけ。
- Web 向けのビルドは `web` という cargo の profile を使う（小ささを優先し、strip しない）。
- 環境によって効き方の変わる最適化は増やさない。ループの書き下しは、外せる段としてだけ持つ。
- mikage は軽い描画の枠組みとして使う。OS の機能や、呼び出し側でできることを mikage に求めない。

## コマンド

- `cargo run --release`：アプリ（V ビュー / H パネルを隠す・出す / P パネルの上下）。
- `SP_AGENT=1 cargo run --release`：操作口（HTTP、127.0.0.1:3939）つき。`/screenshot` で窓の絵が取れる。
  最初の 1 枚は egui が大きさを決める前なので、起動して 1 秒ほど待ってから撮る。
- `mise run verify`：fmt・clippy・テスト（GPU は使わない。テストは WGSL の全プログラムを naga に通す）。
- `cargo run --release --example run -- OUT --steps N [--dump A,B,..] [--light EVERY] [--load FOLDER,STEP] [--turn A,B,..]`：
  画面なしで模型を走らせ、指定した更新のあとのバッファを `OUT/step-NNNNN/` に書く（GPU を使う）。`--steps 0` で
  1 更新の段階と step の数だけを出す。
- `trunk build` / `mise run web`：Web 向けのビルドと、http://127.0.0.1:8080 での配信。
