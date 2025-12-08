"""
Upgraded APKDecompiler for XAIDroid
- Multi-dex support (classes.dex, classes2.dex, ... and embedded .dex/.jar inside apk)
- Parses smali instructions to extract invoke-* calls (fallback to XREF)
- Detects reflection usage (Class.forName, getMethod, Method.invoke patterns)
- Detects JNI/native usage (native methods, presence of .so libs, System.loadLibrary)
- Handles apkm/apks/zip containers, extracts base.apk or largest .apk
- Produces a rich result dict with extra fields:
    {
      "apk_name": ...,
      "package_name": ...,
      "methods": [...],
      "classes": {...},
      "smali_code": {...},
      "api_calls": {...},
      "embedded_files": [...],
      "native_libs": [...],
      "reflection_evidence": {...},
      "success": True/False,
      "error": "..."
    }
"""
import logging
import tempfile
import shutil
import zipfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from androguard.core.bytecodes.apk import APK
from androguard.core.bytecodes.dvm import DalvikVMFormat
from androguard.core.analysis.analysis import Analysis

logger = logging.getLogger(__name__)


def _is_zip(path: Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except Exception:
        return False


def _extract_base_apk_from_archive(path: Path) -> Optional[Path]:
    """
    If the file is a zip-like archive containing .apk files (APKMirror style),
    extract base.apk if present, otherwise extract the largest .apk and return its path.
    Returns None on failure (caller should fall back to original file).
    """
    try:
        if not _is_zip(path):
            return None

        tmpdir = Path(tempfile.mkdtemp(prefix="xaidroid_apk_"))
        with zipfile.ZipFile(path, "r") as z:
            members = z.namelist()
            apk_members = [m for m in members if m.lower().endswith(".apk")]
            if not apk_members:
                shutil.rmtree(tmpdir, ignore_errors=True)
                return None

            chosen = None
            if "base.apk" in members:
                chosen = "base.apk"
            elif len(apk_members) == 1:
                chosen = apk_members[0]
            else:
                # pick largest .apk inside
                largest = None
                largest_size = -1
                for m in apk_members:
                    info = z.getinfo(m)
                    if info.file_size > largest_size:
                        largest = m
                        largest_size = info.file_size
                chosen = largest

            out_path = tmpdir / "extracted_base.apk"
            with z.open(chosen) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            return out_path
    except Exception as e:
        logger.debug(f"Archive extraction failed for {path.name}: {e}", exc_info=True)
        return None


def _safe_get_dex_blobs(apk_obj: APK, apk_file_path: Path) -> List[Tuple[str, bytes]]:
    """
    Return list of (name, raw dex bytes).
    - Use androguard getters if available
    - Also scan the zip entries inside the APK for .dex/.jar
    - Filter out invalid/too-small entries
    """
    dex_items: List[Tuple[str, bytes]] = []

    # 1) Androguard getters (get_all_dex / get_dex)
    for getter in ("get_all_dex", "get_dex"):
        if hasattr(apk_obj, getter):
            try:
                raw = getattr(apk_obj, getter)()
                if raw is None:
                    continue
                # raw could be generator, list or single bytes
                if isinstance(raw, (bytes, bytearray)):
                    raw_list = [bytes(raw)]
                    names = ["classes.dex"]
                else:
                    raw_list = list(raw)
                    names = [f"classes{i+1}.dex" for i in range(len(raw_list))]
                for name, item in zip(names, raw_list):
                    if isinstance(item, (bytes, bytearray)) and len(item) >= 4:
                        dex_items.append((name, bytes(item)))
            except Exception:
                logger.debug(f"Error while calling {getter} on APK object", exc_info=True)
                continue

    # 2) Manual scan of zip entries for embedded .dex / .jar
    try:
        with zipfile.ZipFile(apk_file_path, 'r') as z:
            for info in z.infolist():
                nm = info.filename
                if nm.lower().endswith(".dex") or nm.lower().endswith(".jar"):
                    try:
                        with z.open(nm) as f:
                            data = f.read()
                            if data and len(data) >= 4:
                                dex_items.append((nm, data))
                    except Exception:
                        logger.debug(f"Failed to read embedded file {nm} in {apk_file_path.name}", exc_info=True)
                        continue
    except Exception:
        logger.debug("Failed to open APK as zip for embedded file scan", exc_info=True)

    # Remove duplicates by name keeping first occurrence
    seen: Set[str] = set()
    unique: List[Tuple[str, bytes]] = []
    for name, data in dex_items:
        if name not in seen:
            seen.add(name)
            unique.append((name, data))

    return unique


class APKDecompiler:
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # regex to find invoke-* style called signatures in smali output
        # e.g. "invoke-virtual {p0}, Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;"
        self._invoke_regex = re.compile(r'(L[^;\s]+;->\w+\(.*?\).+?)\b')
        # const-string pattern to capture class/method names or strings used before reflection
        self._const_string_regex = re.compile(r'const-string(?:/jumbo)?\s+.*?,\s+"([^"]+)"')
        # pattern for System.loadLibrary or Class.forName usage in smali
        self._forname_regex = re.compile(r'Ljava/lang/Class;->forName|Ljava/lang/reflect/Method;->invoke|System->loadLibrary')

    def decompile(self, apk_path: str) -> Dict:
        apk_path = Path(apk_path)
        apk_name = apk_path.name
        logger.info(f"Decompiling APK: {apk_name}")

        extracted_tmp = None
        try:
            extracted = _extract_base_apk_from_archive(apk_path)
            if extracted:
                logger.info(f"Extracted inner APK from {apk_name} -> {extracted.name}")
                apk_to_use = extracted
                extracted_tmp = extracted.parent
            else:
                apk_to_use = apk_path
        except Exception as e:
            logger.warning(f"Failed to extract from archive {apk_name}: {e}")
            apk_to_use = apk_path

        try:
            # Basic zip validity
            try:
                if not _is_zip(apk_to_use):
                    return {"success": False, "error": "Not a valid APK/ZIP archive"}
            except Exception:
                return {"success": False, "error": "Failed to validate APK file"}

            # Load via androguard
            try:
                apk_obj = APK(str(apk_to_use))
            except Exception as e:
                return {"success": False, "error": f"Androguard APK load failed: {e}"}

            # Validate manifest
            try:
                manifest_bytes = apk_obj.get_android_manifest_xml()
            except Exception:
                manifest_bytes = None

            if not manifest_bytes:
                # keep trying — sometimes the manifest is accessible via APK api but empty; we still allow but warn
                logger.warning(f"Missing AndroidManifest.xml in {apk_name}")
                # Instead of failing immediately, continue but mark it
                manifest_ok = False
            else:
                manifest_ok = True

            try:
                pkg = apk_obj.get_package()
            except Exception:
                pkg = None

            # Collect embedded files info (so, dex, jars)
            embedded_files = []
            native_libs = []
            try:
                with zipfile.ZipFile(apk_to_use, 'r') as z:
                    for info in z.infolist():
                        nm = info.filename
                        if nm.lower().endswith(".so"):
                            native_libs.append(nm)
                        if nm.lower().endswith(".dex") or nm.lower().endswith(".jar"):
                            embedded_files.append({"name": nm, "size": info.file_size})
            except Exception:
                logger.debug("Failed to enumerate embedded files", exc_info=True)

            # Get dex blobs (name, bytes)
            dex_items = _safe_get_dex_blobs(apk_obj, apk_to_use)
            if not dex_items:
                # If we have manifest missing or no dex, return informative error
                msg = "No DEX files found"
                logger.warning(f"{apk_name}: {msg}")
                return {"success": False, "error": msg}

            # Build DalvikVMFormat objects, guard corrupt blobs
            dalvik_objs = []
            for name, raw in dex_items:
                try:
                    d = DalvikVMFormat(raw)
                    dalvik_objs.append((name, d))
                except Exception as e:
                    logger.debug(f"Skipping corrupt dex blob {name} in {apk_name}: {e}", exc_info=True)

            if not dalvik_objs:
                return {"success": False, "error": "All DEX blobs invalid or corrupt"}

            # Analysis aggregator
            analysis = Analysis()
            for _, d in dalvik_objs:
                try:
                    analysis.add(d)
                except Exception as e:
                    logger.debug(f"Failed to add dex to analysis for {apk_name}: {e}", exc_info=True)
            try:
                analysis.create_xref()
            except Exception:
                logger.debug(f"create_xref failed for {apk_name}", exc_info=True)

            # Prepare result
            result = {
                "apk_name": apk_name,
                "package_name": pkg,
                "methods": [],
                "classes": {},
                "smali_code": {},
                "api_calls": {},
                "embedded_files": embedded_files,
                "native_libs": native_libs,
                "reflection_evidence": {},
                "success": True,
                "manifest_ok": manifest_ok
            }

            # Walk each dex and extract
            for dex_name, d in dalvik_objs:
                self._extract_methods(d, analysis, result, dex_name)

            logger.info(f"Decompiled {apk_name}: {len(result['methods'])} methods, {len(result['classes'])} classes, embedded_files={len(embedded_files)}, native_libs={len(native_libs)}")

            return result

        except Exception as e:
            logger.error(f"Unhandled error while decompiling {apk_name}: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
        finally:
            if extracted_tmp and extracted_tmp.exists():
                try:
                    shutil.rmtree(extracted_tmp, ignore_errors=True)
                except Exception:
                    pass

    def _extract_methods(self, dex: DalvikVMFormat, analysis: Analysis, result: Dict, dex_name: str) -> None:
        """
        Populate result dict with methods/classes/smali/api_calls for a single DalvikVMFormat.
        """
        try:
            classes = dex.get_classes()
        except Exception:
            logger.debug("dex.get_classes() failed; skipping this dex", exc_info=True)
            return

        for cls in classes:
            try:
                class_name = cls.get_name()
            except Exception:
                continue

            if self._is_framework_class(class_name):
                continue

            if class_name not in result["classes"]:
                result["classes"][class_name] = []

            try:
                methods = cls.get_methods()
            except Exception:
                methods = []

            for method in methods:
                try:
                    method_name = method.get_name()
                    method_desc = method.get_descriptor()
                    full_name = f"{class_name}->{method_name}{method_desc}"
                except Exception:
                    continue

                info = {
                    "name": full_name,
                    "class": class_name,
                    "access_flags": None,
                    "descriptor": method_desc,
                    "code_size": 0,
                    "dex": dex_name
                }

                try:
                    info["access_flags"] = method.get_access_flags_string()
                except Exception:
                    info["access_flags"] = None

                # safe code size
                try:
                    code_obj = method.get_code()
                    if code_obj is not None:
                        ins = code_obj.get_bc().get_instructions()
                        try:
                            info["code_size"] = len(list(ins))
                        except Exception:
                            count = 0
                            try:
                                for _ in ins:
                                    count += 1
                            except Exception:
                                count = 0
                            info["code_size"] = count
                except Exception:
                    info["code_size"] = 0

                result["methods"].append(info)
                result["classes"][class_name].append(full_name)

                # smali-like extraction
                try:
                    smali_lines = self._extract_smali_code(method)
                    result["smali_code"][full_name] = smali_lines
                except Exception:
                    result["smali_code"][full_name] = []

                # api calls extraction: first try xref, fallback to smali parsing
                try:
                    calls, reflection_notes = self._extract_api_calls(method, analysis, result["smali_code"].get(full_name, []))
                    # deduplicate
                    unique_calls = []
                    seen_calls = set()
                    for c in calls:
                        if c not in seen_calls:
                            seen_calls.add(c)
                            unique_calls.append(c)
                    result["api_calls"][full_name] = unique_calls
                    if reflection_notes:
                        result["reflection_evidence"].setdefault(full_name, []).extend(reflection_notes)
                except Exception:
                    result["api_calls"][full_name] = []

    def _extract_smali_code(self, method) -> List[str]:
        lines = []
        try:
            code_obj = method.get_code()
            if code_obj is None:
                return lines
            ins_iter = code_obj.get_bc().get_instructions()
            for ins in ins_iter:
                try:
                    out = ins.get_output()
                    if out:
                        lines.append(out)
                except Exception:
                    continue
        except Exception:
            pass
        return lines

    def _extract_api_calls(self, method, analysis: Analysis, smali_lines: List[str]) -> Tuple[List[str], List[str]]:
        """
        Extract API calls for a method. Strategy:
         1) Try androguard xref -> get_xref_to()
         2) If xref empty, parse smali lines for invoke-* patterns
         3) Detect reflection sequences via const-string + forName/getMethod/invoke
         4) Detect native methods via method access flags or JNI presence
        Returns (list_of_calls, reflection_notes)
        """
        api_calls: List[str] = []
        reflection_notes: List[str] = []

        # 1) XREF-based extraction (preferred)
        x_method = None
        try:
            x_method = analysis.get_method(method)
        except Exception:
            x_method = None

        if x_method:
            try:
                for _, call, _ in x_method.get_xref_to():
                    try:
                        called = call.get_method()
                        called_cls = called.get_class_name()
                        called_sig = f"{called_cls}->{called.get_name()}{called.get_descriptor()}"
                        api_calls.append(called_sig)
                    except Exception:
                        continue
            except Exception:
                # If xref iteration fails, we'll fallback
                pass

        # 2) Fallback: parse smali lines for invoke patterns
        if not api_calls and smali_lines:
            for line in smali_lines:
                # try to find explicit 'L...;->method(...)...' patterns
                for m in self._invoke_regex.finditer(line):
                    try:
                        called_sig = m.group(1).strip()
                        api_calls.append(called_sig)
                    except Exception:
                        continue

                # Also capture calls like 'invoke-static {v0}, Ljava/lang/Class;->forName(Ljava/lang/String;)Ljava/lang/Class;'
                if 'forName' in line or 'Ljava/lang/reflect/Method' in line or 'System->loadLibrary' in line:
                    # mark reflection/jni evidence
                    reflection_notes.append(f"pattern:{line.strip()}")

        # 3) Additional heuristic: reconstruct reflective call targets using nearby const-strings
        # Look for sequences: const-string "com.example.BadClass" ... invoke-static ... forName ... invoke-virtual getMethod/invoke
        # We'll scan smali_lines for const-strings in the same method
        const_strings = []
        for sline in smali_lines:
            cs = self._const_string_regex.search(sline)
            if cs:
                const_strings.append(cs.group(1))

        if const_strings:
            # attempt to correlate forName/getMethod/invoke occurrences
            for i, sline in enumerate(smali_lines):
                if 'forName(' in sline or 'Ljava/lang/Class;->forName' in sline:
                    # try to find preceding const-string (simple heuristic)
                    window = smali_lines[max(0, i-6):i+6]
                    found_classname = None
                    for w in reversed(window):
                        m = self._const_string_regex.search(w)
                        if m:
                            found_classname = m.group(1)
                            break
                    if found_classname:
                        reflection_notes.append(f"forName detected -> class='{found_classname}'")
                        # try to find subsequent getMethod + invoke
                        for later in smali_lines[i:i+12]:
                            if 'getMethod(' in later or 'Ljava/lang/reflect/Method;->getMethod' in later:
                                reflection_notes.append(f"getMethod detected after forName in method; possible reflective call on {found_classname}")
                            if 'invoke(' in later or 'Ljava/lang/reflect/Method;->invoke' in later:
                                reflection_notes.append(f"Method.invoke detected -> runtime execution of {found_classname}")

        # 4) Native/JNI detection
        try:
            acc = method.get_access_flags_string() if hasattr(method, 'get_access_flags_string') else None
            if acc and 'native' in (acc.lower() if isinstance(acc, str) else ''):
                reflection_notes.append("method marked native")
        except Exception:
            pass

        # 5) In-method detection for System.loadLibrary or JNI patterns in smali
        for line in smali_lines:
            if 'System.loadLibrary' in line or 'Ljava/lang/System;->loadLibrary' in line:
                reflection_notes.append("System.loadLibrary called (loads native library)")
            if 'JNI_OnLoad' in line:
                reflection_notes.append("JNI_OnLoad detected")

        return api_calls, reflection_notes

    def _is_framework_class(self, cls_name: str) -> bool:
        if not cls_name:
            return False
        prefixes = (
            "Landroid/", "Ljava/", "Ljavax/", "Ldalvik/",
            "Lorg/apache/", "Lorg/json/", "Lorg/xml/", "Lorg/w3c/"
        )
        return any(cls_name.startswith(p) for p in prefixes)
