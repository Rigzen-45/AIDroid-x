"""
Smali Parser Module
Parses Smali bytecode to extract methods, instructions, and API calls
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SmaliMethod:
    """Represents a method in Smali code."""
    class_name: str
    method_name: str
    descriptor: str
    access_flags: List[str]
    instructions: List[str]
    api_calls: List[str]
    registers: int
    parameters: List[str]


@dataclass
class SmaliClass:
    """Represents a class in Smali code."""
    class_name: str
    super_class: str
    interfaces: List[str]
    access_flags: List[str]
    methods: List[SmaliMethod]
    fields: List[str]


class SmaliParser:
    """
    Parser for Smali bytecode files.
    Extracts class structure, methods, and API calls.
    """
    
    def __init__(self):
        """Initialize Smali parser."""
        # Regex patterns for Smali syntax
        self.class_pattern = re.compile(r'^\.class\s+(.*?)\s+(L.+;)$')
        self.super_pattern = re.compile(r'^\.super\s+(L.+;)$')
        self.interface_pattern = re.compile(r'^\.implements\s+(L.+;)$')
        self.method_pattern = re.compile(r'^\.method\s+(.*?)\s+(.+?)\((.+?)\)(.+?)$')
        self.end_method_pattern = re.compile(r'^\.end method$')
        self.invoke_pattern = re.compile(
            r'^invoke-\w+(?:/range)?\s+(?:\{.*?\}|v\d+(?:\.\.v\d+)?),\s*(.+?)->(.+?)\((.+?)\)(.+?)$'
        )
        self.field_pattern = re.compile(r'^\.field\s+(.*?)\s+(.+?):(.+?)$')
        self.registers_pattern = re.compile(r'^\.registers\s+(\d+)$')
        self.locals_pattern = re.compile(r'^\.locals\s+(\d+)$')
    
    def parse_file(self, smali_file_path: str) -> Optional[SmaliClass]:
        """
        Parse a single Smali file.
        
        Args:
            smali_file_path: Path to .smali file
            
        Returns:
            SmaliClass object or None if parsing fails
        """
        try:
            with open(smali_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            return self.parse_lines(lines)
            
        except Exception as e:
            logger.error(f"Error parsing {smali_file_path}: {e}")
            return None
    
    def parse_lines(self, lines: List[str]) -> Optional[SmaliClass]:
        """
        Parse Smali code lines.
        
        Args:
            lines: List of Smali code lines
            
        Returns:
            SmaliClass object or None
        """
        class_name = None
        super_class = None
        interfaces = []
        access_flags = []
        methods = []
        fields = []
        
        current_method = None
        method_instructions = []
        method_api_calls = []
        method_registers = 0
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Parse class declaration
            class_match = self.class_pattern.match(line)
            if class_match:
                access_flags = class_match.group(1).split()
                class_name = class_match.group(2)
                i += 1
                continue
            
            # Parse super class
            super_match = self.super_pattern.match(line)
            if super_match:
                super_class = super_match.group(1)
                i += 1
                continue
            
            # Parse interfaces
            interface_match = self.interface_pattern.match(line)
            if interface_match:
                interfaces.append(interface_match.group(1))
                i += 1
                continue
            
            # Parse fields
            field_match = self.field_pattern.match(line)
            if field_match:
                fields.append(line)
                i += 1
                continue
            
            # Parse method start
            method_match = self.method_pattern.match(line)
            if method_match:
                method_access_flags = method_match.group(1).split()
                method_name = method_match.group(2)
                method_params = method_match.group(3)
                method_return = method_match.group(4)
                
                method_descriptor = f"({method_params}){method_return}"
                
                current_method = {
                    'access_flags': method_access_flags,
                    'name': method_name,
                    'descriptor': method_descriptor,
                    'parameters': self._parse_parameters(method_params)
                }
                method_instructions = []
                method_api_calls = []
                method_registers = 0
                i += 1
                continue
            
            # Parse method end
            if self.end_method_pattern.match(line):
                if current_method:
                    method = SmaliMethod(
                        class_name=class_name,
                        method_name=current_method['name'],
                        descriptor=current_method['descriptor'],
                        access_flags=current_method['access_flags'],
                        instructions=method_instructions,
                        api_calls=method_api_calls,
                        registers=method_registers,
                        parameters=current_method['parameters']
                    )
                    methods.append(method)
                    current_method = None
                i += 1
                continue
            
            # Inside method body
            if current_method:
                # Parse registers
                registers_match = self.registers_pattern.match(line)
                if registers_match:
                    method_registers = int(registers_match.group(1))
                
                locals_match = self.locals_pattern.match(line)
                if locals_match:
                    method_registers = int(locals_match.group(1))
                
                # Parse invoke instructions (API calls)
                invoke_match = self.invoke_pattern.match(line)
                if invoke_match:
                    target_class = invoke_match.group(1)
                    target_method = invoke_match.group(2)
                    target_params = invoke_match.group(3)
                    target_return = invoke_match.group(4)
                    
                    api_call = f"{target_class}->{target_method}({target_params}){target_return}"
                    method_api_calls.append(api_call)
                
                # Store all instructions
                method_instructions.append(line)
            
            i += 1
        
        # Create SmaliClass object
        if class_name:
            smali_class = SmaliClass(
                class_name=class_name,
                super_class=super_class or "Ljava/lang/Object;",
                interfaces=interfaces,
                access_flags=access_flags,
                methods=methods,
                fields=fields
            )
            return smali_class
        
        return None
    
    def _parse_parameters(self, param_string: str) -> List[str]:
        """
        Parse method parameter types from descriptor.
        
        Args:
            param_string: Parameter string like "Ljava/lang/String;I"
            
        Returns:
            List of parameter types
        """
        if not param_string:
            return []
        
        params = []
        i = 0
        current_param = ""
        
        while i < len(param_string):
            char = param_string[i]
            
            if char == 'L':
                # Object type - read until semicolon
                j = i
                while j < len(param_string) and param_string[j] != ';':
                    j += 1
                params.append(param_string[i:j+1])
                i = j + 1
            elif char == '[':
                # Array type - read next type
                current_param = char
                i += 1
            elif char in 'ZBCSIJFD':
                # Primitive type
                params.append(current_param + char)
                current_param = ""
                i += 1
            else:
                i += 1
        
        return params
    
    def parse_directory(self, smali_dir: str) -> List[SmaliClass]:
        """
        Parse all Smali files in a directory.
        
        Args:
            smali_dir: Directory containing .smali files
            
        Returns:
            List of SmaliClass objects
        """
        smali_dir = Path(smali_dir)
        smali_files = list(smali_dir.rglob("*.smali"))
        
        logger.info(f"Found {len(smali_files)} Smali files in {smali_dir}")
        
        classes = []
        for smali_file in smali_files:
            smali_class = self.parse_file(str(smali_file))
            if smali_class:
                classes.append(smali_class)
        
        logger.info(f"Successfully parsed {len(classes)} classes")
        return classes
    
    def extract_api_calls(self, smali_class: SmaliClass) -> Dict[str, List[str]]:
        """
        Extract all API calls from a Smali class.
        
        Args:
            smali_class: SmaliClass object
            
        Returns:
            Dictionary mapping method signatures to API calls
        """
        api_calls = {}
        
        for method in smali_class.methods:
            method_signature = f"{smali_class.class_name}->{method.method_name}{method.descriptor}"
            api_calls[method_signature] = method.api_calls
        
        return api_calls
    
    def get_statistics(self, smali_class: SmaliClass) -> Dict:
        """
        Get statistics about a Smali class.
        
        Args:
            smali_class: SmaliClass object
            
        Returns:
            Dictionary of statistics
        """
        total_instructions = sum(len(m.instructions) for m in smali_class.methods)
        total_api_calls = sum(len(m.api_calls) for m in smali_class.methods)
        
        return {
            'class_name': smali_class.class_name,
            'num_methods': len(smali_class.methods),
            'num_fields': len(smali_class.fields),
            'total_instructions': total_instructions,
            'total_api_calls': total_api_calls,
            'avg_instructions_per_method': total_instructions / len(smali_class.methods) if smali_class.methods else 0,
            'avg_api_calls_per_method': total_api_calls / len(smali_class.methods) if smali_class.methods else 0
        }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with sample Smali code
    sample_smali = """
.class public Lcom/example/MainActivity;
.super Landroid/app/Activity;

.method public onCreate(Landroid/os/Bundle;)V
    .registers 3
    
    invoke-super {p0, p1}, Landroid/app/Activity;->onCreate(Landroid/os/Bundle;)V
    
    invoke-virtual {p0}, Lcom/example/MainActivity;->getDeviceId()Ljava/lang/String;
    
    return-void
.end method

.method private getDeviceId()Ljava/lang/String;
    .registers 3
    
    const-string v0, "phone"
    
    invoke-virtual {p0, v0}, Lcom/example/MainActivity;->getSystemService(Ljava/lang/String;)Ljava/lang/Object;
    move-result-object v1
    
    check-cast v1, Landroid/telephony/TelephonyManager;
    
    invoke-virtual {v1}, Landroid/telephony/TelephonyManager;->getDeviceId()Ljava/lang/String;
    move-result-object v0
    
    return-object v0
.end method
""".strip().split('\n')
    
    parser = SmaliParser()
    smali_class = parser.parse_lines(sample_smali)
    
    if smali_class:
        print(f"Class: {smali_class.class_name}")
        print(f"Super: {smali_class.super_class}")
        print(f"Methods: {len(smali_class.methods)}")
        
        for method in smali_class.methods:
            print(f"\nMethod: {method.method_name}{method.descriptor}")
            print(f"  API Calls: {len(method.api_calls)}")
            for api_call in method.api_calls:
                print(f"    - {api_call}")
        
        stats = parser.get_statistics(smali_class)
        print(f"\nStatistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")