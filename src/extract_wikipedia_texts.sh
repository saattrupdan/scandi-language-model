#!/bin/sh
#
# Extract and clean a raw compressed Wikipedia dump.
#
# Usage:
#     sh extract_wikipedia_texts.sh <dump_file>

set -e

WIKI_DUMP_FILE_IN=$1
WIKI_DUMP_FILE_OUT=${WIKI_DUMP_FILE_IN%%.*}.txt

# extract and clean the chosen Wikipedia dump
echo "Extracting and cleaning $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT..."
wikiextractor $WIKI_DUMP_FILE_IN --processes 8 -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT
echo "Succesfully extracted and cleaned $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT"
