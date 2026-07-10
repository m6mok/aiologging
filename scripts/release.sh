#!/usr/bin/env bash
# Единая точка релиза (make release).
#
# Формат релиза (см. docs/releases.md):
#   тэг        vX.Y.Z
#   заголовок  vX.Y.Z (совпадает с тэгом)
#   заметки    секция "### X.Y.Z" из README.md (Changelog)
#
# Шаги: предпроверки -> локальный гейт -> push -> зелёный CI ->
# gh release create (публикация триггерит release.yml -> PyPI).
set -euo pipefail

die() { echo "release: $*" >&2; exit 1; }

command -v gh >/dev/null || die "нужен gh (GitHub CLI)"
gh auth status >/dev/null 2>&1 || die "gh не авторизован (gh auth login)"

# --- предпроверки -----------------------------------------------------
[ -z "$(git status --porcelain)" ] \
    || die "рабочее дерево не чистое — закоммитьте или отложите изменения"
branch=$(git rev-parse --abbrev-ref HEAD)
[ "$branch" = "master" ] || die "релиз только с master (сейчас: $branch)"

version=$(sed -n 's/^version = "\(.*\)"/\1/p' pyproject.toml | head -1)
initver=$(sed -n 's/^__version__ = "\(.*\)"/\1/p' aiologging/__init__.py)
[ -n "$version" ] || die "не нашёл version в pyproject.toml"
[ "$version" = "$initver" ] \
    || die "версии рассинхронизированы: pyproject=$version __init__=$initver"

tag="v$version"
git rev-parse -q --verify "refs/tags/$tag" >/dev/null \
    && die "тэг $tag уже существует локально"
[ -z "$(git ls-remote --tags origin "refs/tags/$tag")" ] \
    || die "тэг $tag уже существует на origin"
gh release view "$tag" >/dev/null 2>&1 \
    && die "релиз $tag уже опубликован"

grep -qxF "### $version" README.md \
    || die "в README.md нет секции changelog '### $version'"
notes=$(awk -v v="### $version" \
    '$0 == v {f=1; next} /^### / {f=0} f' README.md)
[ -n "$(echo "$notes" | tr -d '[:space:]')" ] \
    || die "секция changelog для $version пуста"

# --- локальный гейт ---------------------------------------------------
make quick-test quick-mypy stress-quick

# --- push и ожидание CI -----------------------------------------------
git push origin master
sha=$(git rev-parse HEAD)
echo "release: жду появления CI-рана для $sha ..."
run_id=""
for _ in $(seq 1 30); do
    run_id=$(gh run list --commit "$sha" --limit 1 \
        --json databaseId --jq '.[0].databaseId' 2>/dev/null || true)
    [ -n "$run_id" ] && break
    sleep 5
done
[ -n "$run_id" ] || die "CI-ран для $sha так и не появился"
gh run watch "$run_id" --exit-status \
    || die "CI красный — релиз не создан"

# --- публикация -------------------------------------------------------
gh release create "$tag" --title "$tag" --notes "$notes"
echo "release: $tag опубликован; загрузку на PyPI смотрите так:"
echo "  gh run list --workflow=release.yml --limit 1"
