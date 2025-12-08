"""
Upgraded API Extractor and Sensitive API Filter for XAIDroid
- Works with upgraded APKDecompiler (multi-dex + smali + reflection evidence)
- Normalizes smali/androguard API signatures and applies layered matching
- Maps reflection evidence to sensitive API candidates
- No external dependencies (pure stdlib + same JSON sensitive db)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import math

logger = logging.getLogger(__name__)


class SensitiveAPIFilter:
    def __init__(self, sensitive_api_path: str = "config/sensitive_apis.json",
                 aggressive_matching: bool = False):
        """
        Args:
            sensitive_api_path: path to JSON sensitive API database.
            aggressive_matching: if True, method-only matches are treated as sensitive immediately.
        """
        self.sensitive_api_path = Path(sensitive_api_path)
        self.sensitive_apis: Set[str] = set()
        self.api_to_category = {}
        self.categories = {}
        self.category_to_id = {}
        self.aggressive_matching = aggressive_matching

        self._load_sensitive_apis()
        logger.info(f"SensitiveAPIFilter loaded {len(self.sensitive_apis)} APIs across {len(self.categories)} categories")

    # --------------------------
    # Loading & helpers
    # --------------------------
    def _load_sensitive_apis(self) -> None:
        if not self.sensitive_api_path.exists():
            raise FileNotFoundError(f"Sensitive API DB not found: {self.sensitive_api_path}")
        with open(self.sensitive_api_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cat_id = 0
        for cname, cdata in data.get("categories", {}).items():
            apis = cdata.get("apis", [])
            self.categories[cname] = apis
            self.category_to_id[cname] = cat_id
            cat_id += 1
            for api in apis:
                norm_api = self._normalize_sig(api)
                self.sensitive_apis.add(norm_api)
                self.api_to_category[norm_api] = cname

    def _normalize_sig(self, api_call: str) -> str:
        """
        Normalize API signature to canonical smali-like form:
          - Strip whitespace
          - Ensure class uses L...; format (if dot-style passed, convert)
          - Remove return type suffix when present (keep descriptor parentheses)
        Examples normalized:
          'Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;'
          -> 'Landroid/telephony/TelephonyManager;->getDeviceId()'
        """
        if not api_call:
            return ""
        s = api_call.strip()

        # if dot-style like android.telephony.TelephonyManager.getDeviceId -> convert
        if "->" not in s and "." in s:
            parts = s.split(".")
            cls = "L" + "/".join(parts[:-1]) + ";"
            method = parts[-1] + "()"
            s = f"{cls}->{method}"

        # If contains '->' and parameters and return types, keep up to closing ')'
        if "->" in s:
            try:
                left, right = s.split("->", 1)
                if "(" in right:
                    method_part = right.split(")")[0] + ")"
                else:
                    method_part = right
                s = f"{left}->{method_part}"
            except Exception:
                s = s.split(" #")[0]
        # Normalize multiple slashes, remove duplicate spaces
        s = s.replace(" ;", ";")
        s = s.replace(" )", ")")
        return s

    def _split_class_method(self, api_call: str) -> Tuple[str, str]:
        """
        Return (class, method_name)
        class e.g. Landroid/telephony/TelephonyManager;
        method_name e.g. getDeviceId
        """
        normalized = self._normalize_sig(api_call)
        if "->" not in normalized:
            return normalized, ""
        cls, method = normalized.split("->", 1)
        method_name = method.split("(")[0]
        return cls, method_name

    def _method_similarity(self, a: str, b: str) -> float:
        """
        Simple similarity for method names (0..1)
        Uses longest common subsequence-based heuristics (fast).
        """
        if not a or not b:
            return 0.0
        a = a.lower(); b = b.lower()
        if a == b:
            return 1.0
        # quick checks
        if a in b or b in a:
            return 0.9
        # LCS length / max len
        la, lb = len(a), len(b)
        # DP for LCS (small strings = method names)
        dp = [[0]*(lb+1) for _ in range(la+1)]
        for i in range(la-1, -1, -1):
            for j in range(lb-1, -1, -1):
                if a[i] == b[j]:
                    dp[i][j] = 1 + dp[i+1][j+1]
                else:
                    dp[i][j] = max(dp[i+1][j], dp[i][j+1])
        lcs = dp[0][0]
        return lcs / max(la, lb)

    def _class_suffix_match(self, class_a: str, class_b: str) -> bool:
        """
        Match classes by suffix tokens. Useful when packages are shaded/relocated.
        Example: Landroid/telephony/TelephonyManager; vs Lcom/vendor/android/telephony/TelephonyManager;
        """
        if not class_a or not class_b:
            return False
        # strip 'L' and ';'
        a = class_a.strip("L").rstrip(";")
        b = class_b.strip("L").rstrip(";")
        atoks = a.split("/")
        btoks = b.split("/")
        # compare last token (class name) and maybe last 2 tokens
        if atoks[-1] == btoks[-1]:
            return True
        if len(atoks) > 1 and len(btoks) > 1 and "/".join(atoks[-2:]) == "/".join(btoks[-2:]):
            return True
        return False

    # --------------------------
    # Matching logic
    # --------------------------
    def _match_exact(self, api_call: str) -> Optional[str]:
        norm = self._normalize_sig(api_call)
        if norm in self.sensitive_apis:
            return norm
        return None

    def _match_class_level(self, api_call: str) -> Optional[str]:
        cls, _ = self._split_class_method(api_call)
        for sens in self.sensitive_apis:
            scls, _ = self._split_class_method(sens)
            if cls == scls or self._class_suffix_match(cls, scls):
                return sens
        return None

    def _match_method_only(self, api_call: str, threshold: float = 0.95) -> Optional[str]:
        _, m = self._split_class_method(api_call)
        # exact method name
        for sens in self.sensitive_apis:
            _, sm = self._split_class_method(sens)
            if sm == m:
                return sens
        # fuzzy method name
        best = None
        best_score = 0.0
        for sens in self.sensitive_apis:
            _, sm = self._split_class_method(sens)
            score = self._method_similarity(m, sm)
            if score > best_score:
                best_score = score
                best = sens
        if best_score >= threshold:
            return best
        return None

    def is_sensitive(self, api_call: str) -> bool:
        """
        Determine whether the api_call is sensitive using layered matching.
        """
        if not api_call:
            return False

        # 1. exact normalized match
        exact = self._match_exact(api_call)
        if exact:
            return True

        # 2. class-level match (telephony.*, sms.*, location.*)
        cls_match = self._match_class_level(api_call)
        if cls_match:
            # treat as sensitive, but prefer exact if available
            return True

        # 3. method-only match
        method_match = self._match_method_only(api_call)
        if method_match:
            # If aggressive mode is enabled, accept method-only
            if self.aggressive_matching:
                return True
            # otherwise accept if method highly similar
            sim = self._method_similarity(api_call.split("->")[-1].split("(")[0],
                                          method_match.split("->")[-1].split("(")[0])
            return sim >= 0.95

        # 4. fallback fuzzy: compare tokens
        # compare last token of class + method similarity
        try:
            cls, m = self._split_class_method(api_call)
            for sens in self.sensitive_apis:
                scls, sm = self._split_class_method(sens)
                if self._class_suffix_match(cls, scls) and self._method_similarity(m, sm) > 0.8:
                    return True
        except Exception:
            pass

        return False

    # --------------------------
    # Reflection mapping
    # --------------------------
    def extract_reflection_sensitive(self, reflection_notes: List[str]) -> List[str]:
        """
        Convert reflection evidence lines into candidate sensitive APIs.
        It tries to extract class names from notes like:
          "forName detected -> class='android.telephony.SmsManager'"
        and maps to any sensitive APIs with that class.
        """
        results = []
        for note in reflection_notes or []:
            if "class='" in note:
                try:
                    cls = note.split("class='", 1)[1].split("'", 1)[0]
                    dalvik = "L" + cls.replace(".", "/") + ";"
                    # find matching sensitive APIs with that class
                    for sens in self.sensitive_apis:
                        scls, _ = self._split_class_method(sens)
                        if scls == dalvik or self._class_suffix_match(scls, dalvik):
                            results.append(sens)
                except Exception:
                    continue
        return results

    # --------------------------
    # Main filter - accepts either:
    #  - method_api_calls: { method -> [api_sig, ...] }
    #  - OR a decompiler result dict (as returned by APKDecompiler)
    # --------------------------
    def filter_api_calls(self, input_data: Any) -> Dict:
        """
        input_data: either method_api_calls dict or decompiler result dict.

        Returns filtered_data as:
            {
                "methods": { method_name: [sensitive_api1, ...] },
                "api_stats": { api: {count, category, methods: [...] } },
                "category_stats": { cat: count },
                "reflection_hits": { method: [api...] },
                "total_sensitive": int,
                "total_filtered": int
            }
        """
        # Accept either data structure
        method_api_calls: Dict[str, List[str]] = {}
        reflection_map: Dict[str, List[str]] = {}
        # If a decompiler result dict
        if isinstance(input_data, dict) and "methods" in input_data and "api_calls" in input_data:
            # normalized input
            # input_data["api_calls"] is method->list, reflection_evidence maybe present
            method_api_calls = input_data.get("api_calls", {}) or {}
            reflection_map = input_data.get("reflection_evidence", {}) or {}
        else:
            # assume a plain mapping was passed
            method_api_calls = input_data or {}

        # initialize stats
        filtered_data = {
            "methods": {},  # method -> list of sensitive APIs
            "api_stats": {},
            "category_stats": {cat: 0 for cat in self.categories},
            "reflection_hits": {},
            "total_sensitive": 0,
            "total_filtered": 0
        }

        for method_name, api_calls in method_api_calls.items():
            if not isinstance(api_calls, (list, tuple)):
                continue
            sensitive_calls: List[str] = []
            filtered_data["total_filtered"] += len(api_calls)

            # normalize and check each api call
            for raw_call in api_calls:
                norm_call = self._normalize_sig(raw_call)
                # Try to directly match normalized call
                if self.is_sensitive(norm_call):
                    # choose canonical sens api if available
                    matched = self._match_exact(norm_call) or self._match_class_level(norm_call) or self._match_method_only(norm_call) or norm_call
                    sensitive_calls.append(matched)
                    filtered_data["total_sensitive"] += 1
                    # update api_stats
                    api_key = matched
                    if api_key not in filtered_data["api_stats"]:
                        filtered_data["api_stats"][api_key] = {"count": 0, "category": self.api_to_category.get(api_key, "UNKNOWN"), "methods": []}
                    filtered_data["api_stats"][api_key]["count"] += 1
                    filtered_data["api_stats"][api_key]["methods"].append(method_name)
                    # update category
                    cat = filtered_data["api_stats"][api_key]["category"]
                    if cat in filtered_data["category_stats"]:
                        filtered_data["category_stats"][cat] += 1

            # Add reflection-derived hits for this method if any
            refl_notes = reflection_map.get(method_name, []) if reflection_map else []
            refl_candidates = self.extract_reflection_sensitive(refl_notes)
            if refl_candidates:
                # add unique ones
                for rc in refl_candidates:
                    if rc not in sensitive_calls:
                        sensitive_calls.append(rc)
                        filtered_data["total_sensitive"] += 1
                        if rc not in filtered_data["api_stats"]:
                            filtered_data["api_stats"][rc] = {"count": 0, "category": self.api_to_category.get(rc, "UNKNOWN"), "methods": []}
                        filtered_data["api_stats"][rc]["count"] += 1
                        filtered_data["api_stats"][rc]["methods"].append(method_name)
                        cat = filtered_data["api_stats"][rc]["category"]
                        if cat in filtered_data["category_stats"]:
                            filtered_data["category_stats"][cat] += 1
                filtered_data["reflection_hits"][method_name] = refl_candidates

            if sensitive_calls:
                filtered_data["methods"][method_name] = sensitive_calls

        logger.info(f"Filtered {filtered_data['total_sensitive']} sensitive APIs from {filtered_data['total_filtered']} total calls in {len(filtered_data['methods'])} methods")
        return filtered_data

    # --------------------------
    # Utility functions
    # --------------------------
    def get_category(self, api_call: str) -> str:
        norm = self._normalize_sig(api_call)
        return self.api_to_category.get(norm, "UNKNOWN")

    def get_category_id(self, category: str) -> int:
        return self.category_to_id.get(category, -1)

    def get_category_distribution(self, filtered_data: Dict) -> Dict[str, float]:
        total = filtered_data.get("total_sensitive", 0)
        if total == 0:
            return {cat: 0.0 for cat in self.categories}
        return {cat: (count / total) * 100.0 for cat, count in filtered_data.get("category_stats", {}).items()}

    def get_most_common_apis(self, filtered_data: Dict, top_k: int = 10) -> List[Tuple[str, int, str]]:
        api_list = [(api, stats["count"], stats["category"]) for api, stats in filtered_data.get("api_stats", {}).items()]
        api_list.sort(key=lambda x: x[1], reverse=True)
        return api_list[:top_k]

"""
# Example usage (if run directly)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    filt = SensitiveAPIFilter(sensitive_api_path="config/sensitive_apis.json", aggressive_matching=False)

    
    # Example: decompiler result path (the PDF you uploaded is at /mnt/data/COMPLETE WORKFLOW.pdf if you need to reference it)
    # Suppose `decompiler_result` is loaded from your APKDecompiler.decompile()
    decompiler_result_example = {
        "methods": [],
        "api_calls": {
            "Lcom/example/MainActivity;->onCreate()V": [
                "Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;",
                "Landroid/location/LocationManager;->getLastKnownLocation(Ljava/lang/String;)Landroid/location/Location;"
            ],
            "Lcom/obf/a;->a()V": [
                "La/a/a/a;->b()V"  # obfuscated - will be matched by method/class heuristics if possible
            ]
        },
        "reflection_evidence": {
            "Lcom/obf/a;->a()V": ["forName detected -> class='android.telephony.SmsManager'"]
        }
    }
    

    filtered = filt.filter_api_calls(decompiler_result_example) 
    print("Total sensitive:", filtered["total_sensitive"])
    print("Category distribution:", filt.get_category_distribution(filtered))
    print("Top APIs:", filt.get_most_common_apis(filtered, top_k=10))
    """