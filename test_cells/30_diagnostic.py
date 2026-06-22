# ============================================================================
# VERDICT — explicit token-length parity on the two forms + what generated.
# ============================================================================
print("=== token-length parity on the two forms (REAL Llama tokenizer) ===")
all_ok = True
for B, C in [(3, 5), (12, 7), (12, 34), (123, 45), (7, 88)]:
    L = f"( 0 + {B} ) * {C} ="
    R = f"0 + ( {B} * {C} ) ="
    il = tokenizer(L, add_special_tokens=True)["input_ids"]
    ir = tokenizer(R, add_special_tokens=True)["input_ids"]
    ok = (len(il) == len(ir))
    all_ok = all_ok and ok
    print(f"  B={B},C={C}: left={len(il)}tok right={len(ir)}tok  parity={'OK' if ok else 'BROKEN'}")
    if not ok:
        print("    L:", [tokenizer.decode([i]) for i in il])
        print("    R:", [tokenizer.decode([i]) for i in ir])

ps = _MEM.get("phase2_stimuli", [])
print(f"\n=== phase2_stimuli generated: {len(ps)} records ===")
for r in ps[:6]:
    print(f"  {r['condition']:>11} {r['prompt']:<22} = {r['answer']:<7} "
          f"Bidx={r['operand_token_indices']['B']} Cidx={r['operand_token_indices']['C']} "
          f"*idx={r['operator_token_index']} len={r['token_len']}")
if not ps:
    d = _MEM.get("dataset_phase2", {})
    print("  EMPTY. Factor A drop reasons:", d.get("drops", {}).get("A"))

print("\n=== VERDICT ===")
if all_ok and ps:
    print("PASS: parity holds AND the generator produced controlled stimuli on real Llama.")
    print("-> The main notebook's Phase 2 is sound; the earlier empty result was a stale cache /")
    print("   not-run, not a tokenizer fight. In the main notebook, just clear the Drive cache and")
    print("   re-run Phase 2 (the new code self-heals).")
elif not all_ok:
    print("PARITY BROKEN on real Llama -> the *-/paren surface tokenizes asymmetrically.")
    print("-> Paste this output; I'll fix the stimulus surface / restrict operand ranges and push.")
else:
    print("Parity OK but 0 records -> a generator/locator issue on real Llama.")
    print("-> Paste this output; I'll patch the generator.")
