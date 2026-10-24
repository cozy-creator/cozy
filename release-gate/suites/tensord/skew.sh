# Suite tensord, last: version skew degrades, never refuses (owner). An old TensorD (0.11.12, fetched by version) under
# the candidate Runtime and TensorFS runs SDXL standalone, with a warning; run 16438 (Runtime 0.31.0 on 0.11.12) was
# refused memory_protocol_incompatible. Needs the candidate wheels as URLs and an --install-cli old enough to send them
# (cozy 0.11.15: the old TensorD fetches them itself). Sourced from cells.sh with its helpers.
SKEWV=0.11.12
if [ -z "$RW$TW" ] || ! "$ICLI" machine update --help 2>/dev/null | grep -q -- --runtime-wheel || [[ $RW$TW != *://* ]]; then
  log "tensord: skew not run (it needs --runtime/--tensorfs URLs and an --install-cli of cozy 0.11.15 or older)"
else
  "$CLI" machine update "$M" --tensord "$SKEWV" -y > "$T/skew-tensord.log" 2>&1
  "$ICLI" machine update "$M" ${RW:+--runtime-wheel "$RW"} ${TW:+--tensorfs-wheel "$TW"} -y > "$T/skew-wheels.log" 2>&1
  read -r v rt tfs <<< "$(timeout 60 "$CLI" --json machine show "$M" 2>/dev/null | jq -r '.status | "\(.version) \(.runtime) \(.tensorfs)"')"
  n=$(submit "sdxl/generate@$NEW" prompt="$SCENE, an old TensorD" seed:=$((R + 44))); until_status "$n" terminal
  warned=$(grep -hoiE '[^"]*standalone[^"]*' "$T/skew-wheels.log" "$T/$n.follow" "$T/$n.json" 2>/dev/null | head -1)
  if [ "$v" != "$SKEWV" ]; then
    expect tensord-skew FAIL "the machine runs cozy ${v:-?}, not $SKEWV: $(tail -c 200 "$T/skew-tensord.log")"
  elif [ "$(get "$n" status)" = completed ]; then
    expect tensord-skew PASS "TensorD $v under Runtime $rt / TensorFS $tfs: run $n completed${warned:+, warned: $warned}"
  else
    expect tensord-skew FAIL "TensorD $v under Runtime $rt / TensorFS $tfs: run $n $(get "$n" status) $(get "$n" reason)"
  fi
fi
