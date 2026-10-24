# Suite `packages`: only the packages a release changes, the ones named by --pins (and --only), each package's cells
# from products.sh, plus the SDXL smoke. A cell another suite already ran is not run again.
want=" sdxl $(tr , '\n' <<< "$PINS" | cut -d@ -f1 | tr '\n' ' ') $(tr , ' ' <<< "$ONLY") "
for p in "${PRODUCTS[@]}"; do [[ $want != *" $p "* ]] || product "$p"; done
for p in $want; do [[ " ${PRODUCTS[*]} " == *" $p "* ]] || log "packages: $p has no product cells"; done
