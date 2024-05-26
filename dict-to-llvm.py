#!/usr/bin/env python3

import re
import yaml
import sys
import argparse
import jinja2
import logging as log
from typing import Set, Self, Any, List, Tuple
from enum import Enum, IntEnum, auto, unique
from dataclasses import dataclass, asdict


def extract_bits(value: int, offset: int, n: int) -> str:
    decimal = (value >> offset) & ((1 << n) - 1)
    bin = "{0:b}".format(decimal)
    return "0b" + bin.zfill(n)


def is_vector(mnemonic: str) -> bool:
    return mnemonic.startswith(("V", "v"))


def extension_to_decoderns(ext: str) -> str:
    return ext


def extension_to_defprefix(ext: str) -> str:
    return ext.upper() + "_"


def extension_to_encoding_filename(ext: str) -> str:
    return "rv_" + ext.lower()


@unique
class DataType(Enum):
    f8 = "B"
    f8alt = "AB"
    f16 = "H"
    f16alt = "AH"  # bfloat16
    f32 = "S"
    f64 = "D"
    integer = "X"
    uinteger = "XU"
    long = "L"
    ulong = "LU"
    wide = "W"
    uwide = "WU"

    @classmethod
    def from_str(cls, type: str) -> Self:
        for label in cls:
            if type == label.value:
                return cls(label)
        raise ValueError(f"DataType: cannot recognize data type from string: '{type}'")


@unique
class InstructionFormat(IntEnum):
    R = 0  # R-Type
    I = auto()  # I-Type
    S = auto()  # S-Type
    U = auto()  # U-Type
    R4 = auto()  # R4-Type
    RVF = auto()  # V-Type
    # Custom variants of standard formats that rename a known encoding
    # field to match its semantics (e.g.: funct3 -> rm):
    RFRM = auto()
    IFRM = auto()
    R4FRM = auto()
    IIMM12 = auto()
    SIMM12 = auto()
    IVF = auto()

    @classmethod
    def operands(cls):
        # Beware: different formats can have the same set of operands,
        # the first one appearing in the following list will be matched
        return (
            (cls.R, {"rd", "rs1", "rs2"}),
            (cls.I, {"rd", "rs1"}),
            (cls.S, {"rs1", "rs2"}),
            (cls.U, {"rd"}),
            (cls.R4, {"rd", "rs1", "rs2", "rs3"}),
            (cls.RVF, {"rd", "rs1", "rs2"}),  # same as R
            # Variants of standard formats that rename a known encoding
            # field to match its semantics (e.g.: funct3 -> rm):
            (cls.RFRM, {"rd", "rs1", "rs2", "rm"}),
            (cls.IFRM, {"rd", "rs1", "rm"}),
            (cls.R4FRM, {"rs1", "rs2", "rs3", "rd", "rm"}),
            (cls.IIMM12, {"rs1", "rd", "imm12"}),
            (cls.SIMM12, {"rs1", "rs2", "imm12lo", "imm12hi"}),
            (cls.IVF, {"rd", "rs1"}),  # same as I
        )

    @classmethod
    def from_operands(cls, operands: Set[str]) -> Self:
        for fmt, ops in cls.operands():
            if operands == ops:
                return cls(fmt)
        raise ValueError(
            f"InstructionFormat: cannot recognize instruction format from operands: {operands}"
        )


# ENCODING_FIELDS = {
#     "opcode": (0, 7),
#     "rs2": (20, 5),
#     "csr": (20, 12),
#     "funct2": (25, 2),
#     "funct3": (12, 3),
#     "funct7": (25, 7),
#     "imm12": (20, 12),
#     "imm12hi": (25, 7),
#     "imm12lo": (7, 5),
#     "f2": (30, 2),
#     "vecfltop": (25, 5),
#     "r": (14, 1),
#     "vfmt": (12, 2),
# }


@dataclass
class Encoding:

    opcode: str
    rs2: str
    csr: str
    funct2: str
    funct3: str
    funct7: str  # R-Type
    imm12: str  # I-Type
    imm12hi: str  # S-Type
    imm12lo: str  # S-Type
    # Vector fields
    f2: str
    vecfltop: str
    r: str
    vfmt: str

    @classmethod
    def from_int(cls, encoding: int) -> Self:
        return cls(
            opcode=extract_bits(encoding, 0, 7),
            rs2=extract_bits(encoding, 20, 5),
            csr=extract_bits(encoding, 20, 12),
            funct2=extract_bits(encoding, 25, 2),
            funct3=extract_bits(encoding, 12, 3),
            funct7=extract_bits(encoding, 25, 7),
            imm12=extract_bits(encoding, 20, 12),
            imm12hi=extract_bits(encoding, 25, 7),
            imm12lo=extract_bits(encoding, 7, 5),
            # Vector fields
            f2=extract_bits(encoding, 30, 2),
            vecfltop=extract_bits(encoding, 25, 5),
            r=extract_bits(encoding, 14, 1),
            vfmt=extract_bits(encoding, 12, 2),
        )

    @classmethod
    def from_string(cls, encoding: str) -> Self:
        return cls.from_int(int(encoding, 0))


@dataclass
class Instruction:

    mnemonic: str
    encoding: Encoding
    format: InstructionFormat
    encoding_repr: str

    @classmethod
    def from_dict(cls, mnemonic: str, spec: dict) -> Self:
        fmt = InstructionFormat.from_operands(set(spec["variable_fields"]))
        enc = Encoding.from_string(spec["match"])
        # Manually discriminate vector formats
        if is_vector(mnemonic):
            match fmt:
                case InstructionFormat.R:
                    fmt = InstructionFormat.RVF
                case InstructionFormat.I:
                    fmt = InstructionFormat.IVF
                case _:
                    raise RuntimeError(
                        f"Instruction: unknown vector instruction format for {mnemonic}: {fmt}"
                    )

        return cls(
            mnemonic=mnemonic,
            encoding=enc,
            format=fmt,
            encoding_repr=spec["encoding"],
        )


TBLGEN_TEMPLATE_R = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstR<
                {{funct7}}, // funct7
                {{funct3}}, // funct3
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2),
                "{{mnemonic}}", "$rd, $rs1, $rs2">,
                Sched<[]>;
"""

TBLGEN_ALIAS_TEMPLATE_R = """
def : InstAlias<"{{mnemonic}} $rd, $rs1, $rs2", ({{use}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2), 0>;
"""

TBLGEN_TEMPLATE_IIMM12 = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstI<
                {{funct3}}, // funct3
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, simm12:$imm12),
                "{{mnemonic}}", "$rd, $rs1, ${imm12}(${rs1})">,
                Sched<[]>;
"""

TBLGEN_ALIAS_TEMPLATE_IIMM12 = """
def : InstAlias<"{{mnemonic}} $rd, $rs1, $imm12", ({{use}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, simm12:$imm12), 0>;
"""

TBLGEN_TEMPLATE_SIMM12 = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstS<
                {{funct3}}, // funct3
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs ),
                (ins {{dtype["rs2"]}}:$rs2, {{dtype["rs1"]}}:$rs1, simm12:$imm12),
                "{{mnemonic}}", "$rs2, ${imm12}(${rs1})">,
                Sched<[]>;
"""

TBLGEN_ALIAS_TEMPLATE_SIMM12 = """
def : InstAlias<"{{mnemonic}} $rs2, $rs1, $imm12", ({{use}} {{dtype["rs2"]}}:$rs2, {{dtype["rs1"]}}:$rs1, simm12:$imm12), 0>;
"""

TBLGEN_TEMPLATE_R4 = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstR4<
                {{funct2}}, // funct2
                {{funct3}}, // funct3
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rs3"]}}:$rs3),
                "{{mnemonic}}", "$rd, $rs1, $rs2, $rs3">,
                Sched<[]>;
"""

TBLGEN_ALIAS_TEMPLATE_R4 = """
def : InstAlias<"{{mnemonic}} $rd, $rs1, $rs2, $rs3", ({{use}}  {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rs3"]}}:$rs3), 0>;
"""

TBLGEN_TEMPLATE_R4FRM = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstR4Frm<
                {{funct2}}, // funct2
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rs3"]}}:$rs3, frmarg:$frm),
                "{{mnemonic}}", "$rd, $rs1, $rs2, $rs3, $frm">,
                Sched<[]>;
def: InstAlias<"{{mnemonic}} $rd, $rs1, $rs2, $rs3",
               ({{def}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rd"]}}:$rs3, FRM_DYN)>;
"""

TBLGEN_ALIAS_TEMPLATE_R4FRM = """
def : InstAlias<"{{mnemonic}} $rd, $rs1, $rs2, $rs3, frmarg:$frm", ({{use}}  {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rs3"]}}:$rs3, frmarg:$frm), 0>;
"""

TBLGEN_TEMPLATE_RFRM = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstRFrm<
                {{funct7}}, // funct7
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, frmarg:$frm),
                "{{mnemonic}}", "$rd, $rs1, $rs2, $frm">,
                Sched<[]>;
def: InstAlias<"{{mnemonic}} $rd, $rs1, $rs2",
               ({{def}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, FRM_DYN)>;
"""

TBLGEN_ALIAS_TEMPLATE_RFRM = """
def : InstAlias<"{{mnemonic}} $rd, $rs1, $rs2", ({{use}}  {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, FRM_DYN), 0>;
"""

TBLGEN_TEMPLATE_RVF = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstRVf<
                {{f2}}, // f2
                {{vecfltop}}, // vecfltop
                {{r}}, // r
                {{vfmt}}, // vfmt
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2),
                "{{mnemonic}}", "$rd, $rs1, $rs2">,
                Sched<[]>;
"""

TBLGEN_ALIAS_TEMPLATE_RVF = """
def : InstAlias<"{{mnemonic}} $rd, $rs1, $rs2", ({{use}}  {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2), 0>;
"""

TBLGEN_TEMPLATE_IVF = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstRVf<
                {{f2}}, // f2
                {{vecfltop}}, // vecfltop
                {{r}}, // r
                {{vfmt}}, // vfmt
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1),
                "{{mnemonic}}", "$rd, $rs1">,
                Sched<[]>
                { let rs2 = {{rs2}}; }
"""

TBLGEN_ALIAS_TEMPLATE_IVF = """
def : InstAlias<"{{mnemonic}} $rd, $rs1", ({{use}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1), 0>;
"""

TBLGEN_TEMPLATE_IFRM = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstRFrm<
                {{funct7}}, // funct7
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, frmarg:$frm),
                "{{mnemonic}}", "$rd, $rs1, $frm">,
                Sched<[]>
                { let rs2 = {{rs2}}; }
def: InstAlias<"{{mnemonic}} $rd, $rs1",
               ({{def}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, FRM_DYN)>;
"""

TBLGEN_ALIAS_TEMPLATE_IFRM = """
def : InstAlias<"{{mnemonic}} $rd, $rs1", ({{use}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, FRM_DYN), 0>;
"""

TBLGEN_TEMPLATE_I = """
{% if properties -%}
let {% for key, value in properties.items() %}{{key}} = {{value}}{{ ", " if not loop.last else "" }}{% endfor %} in
{%- endif %}
def {{def}} : RVInstI<
                {{funct3}}, // funct3
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1),
                "{{mnemonic}}", "$rd, $rs1">,
                Sched<[]> {
    let imm12 = {{imm12}};
}
"""

TBLGEN_ALIAS_TEMPLATE_I = """
def : InstAlias<"{{mnemonic}} $rd, $rs1", ({{use}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1{% if rm -%}, {{rm}}{%- endif %}), 0>;
"""

LIT_FILE_TEMPLATE = """
# RUN: llvm-mc %s -triple=riscv64 -mattr=+d,+{{ext|lower}} -riscv-no-aliases -show-encoding | FileCheck %s

{% for check in checks %}
# CHECK: encoding: {{check.encoding}}
{{check.asm}}
{% endfor %}
"""

TBLGEN_TEMPLATES = {
    InstructionFormat.R: TBLGEN_TEMPLATE_R,
    InstructionFormat.I: TBLGEN_TEMPLATE_I,
    # InstructionFormat.S: TBLGEN_TEMPLATE_S, // not needed by xsflt
    # InstructionFormat.U: TBLGEN_TEMPLATE_U, // not needed by xsflt
    InstructionFormat.R4: TBLGEN_TEMPLATE_R4,
    InstructionFormat.RFRM: TBLGEN_TEMPLATE_RFRM,
    InstructionFormat.IFRM: TBLGEN_TEMPLATE_IFRM,
    InstructionFormat.R4FRM: TBLGEN_TEMPLATE_R4FRM,
    InstructionFormat.IIMM12: TBLGEN_TEMPLATE_IIMM12,
    InstructionFormat.SIMM12: TBLGEN_TEMPLATE_SIMM12,
    InstructionFormat.RVF: TBLGEN_TEMPLATE_RVF,
    InstructionFormat.IVF: TBLGEN_TEMPLATE_IVF,
}

TBLGEN_ALIAS_TEMPLATES = {
    InstructionFormat.R: TBLGEN_ALIAS_TEMPLATE_R,
    InstructionFormat.I: TBLGEN_ALIAS_TEMPLATE_I,
    # InstructionFormat.S: TBLGEN_ALIAS_TEMPLATE_S, // not needed by xsflt
    # InstructionFormat.U: TBLGEN_ALIAS_TEMPLATE_U, // not needed by xsflt
    InstructionFormat.R4: TBLGEN_ALIAS_TEMPLATE_R4,
    InstructionFormat.RFRM: TBLGEN_ALIAS_TEMPLATE_RFRM,
    InstructionFormat.IFRM: TBLGEN_ALIAS_TEMPLATE_IFRM,
    InstructionFormat.R4FRM: TBLGEN_ALIAS_TEMPLATE_R4FRM,
    InstructionFormat.IIMM12: TBLGEN_ALIAS_TEMPLATE_IIMM12,
    InstructionFormat.SIMM12: TBLGEN_ALIAS_TEMPLATE_SIMM12,
    InstructionFormat.RVF: TBLGEN_ALIAS_TEMPLATE_RVF,
    InstructionFormat.IVF: TBLGEN_ALIAS_TEMPLATE_IVF,
}

TBLGEN_OPERAND_TYPES = {
    DataType.f8: "FPR16",
    DataType.f8alt: "FPR16",
    DataType.f16: "FPR16",
    DataType.f16alt: "FPR16",
    DataType.f32: "FPR32",
    DataType.f64: "FPR64",
    DataType.integer: "GPR",
    DataType.uinteger: "GPR",
    DataType.long: "GPR",
    DataType.ulong: "GPR",
    DataType.wide: "GPR",
    DataType.uwide: "GPR",
}


def get_dtypes(mnemonic: str) -> dict[str, str]:
    # Handle load/stores directly
    match mnemonic:
        case "flh":
            return {
                "rs1": TBLGEN_OPERAND_TYPES[DataType.integer],
                "rd": TBLGEN_OPERAND_TYPES[DataType.f16],
            }
        case "flah":
            return {
                "rs1": TBLGEN_OPERAND_TYPES[DataType.integer],
                "rd": TBLGEN_OPERAND_TYPES[DataType.f16alt],
            }
        case "flb":
            return {
                "rs1": TBLGEN_OPERAND_TYPES[DataType.integer],
                "rd": TBLGEN_OPERAND_TYPES[DataType.f8],
            }
        case "flab":
            return {
                "rs1": TBLGEN_OPERAND_TYPES[DataType.integer],
                "rd": TBLGEN_OPERAND_TYPES[DataType.f8alt],
            }
        case "fsh":
            return {
                "rs1": TBLGEN_OPERAND_TYPES[DataType.integer],
                "rs2": TBLGEN_OPERAND_TYPES[DataType.f16],
            }
        case "fsah":
            return {
                "rs1": TBLGEN_OPERAND_TYPES[DataType.integer],
                "rs2": TBLGEN_OPERAND_TYPES[DataType.f16alt],
            }
        case "fsb":
            return {
                "rs1": TBLGEN_OPERAND_TYPES[DataType.integer],
                "rs2": TBLGEN_OPERAND_TYPES[DataType.f8],
            }
        case "fsab":
            return {
                "rs1": TBLGEN_OPERAND_TYPES[DataType.integer],
                "rs2": TBLGEN_OPERAND_TYPES[DataType.f8alt],
            }

    # Vector format with R: it's just noise in the instruction naming,
    # it doesn't affect operand types, let's remove it
    if is_vector(mnemonic):
        mnemonic = re.sub("[\\._][rR]", "", mnemonic)

    # Reasonable mnemonics
    inst_t = mnemonic.upper().replace(".", "_").split("_")[1:]
    # fcvt.ah.s <- source
    #      ^
    #      dest
    source_t = TBLGEN_OPERAND_TYPES[DataType.from_str(inst_t[-1])]
    dest_t = TBLGEN_OPERAND_TYPES[DataType.from_str(inst_t[0])]

    # FIXME patch logical predicates
    predicates = (
        "fclass",
        "feq",
        "fgt",
        "flt",
        "fle",
        "fge",
    )
    is_predicate = "^(v)?({p})".format(p="|".join(predicates))
    if re.match(is_predicate, mnemonic, re.IGNORECASE):
        dest_t = TBLGEN_OPERAND_TYPES[DataType.integer]

    return {
        "rs1": source_t,
        "rs2": source_t,
        "rs3": source_t,
        "rd": dest_t,
    }


def get_properties(mnemonic: str) -> dict[str, Any]:
    properties = {"hasSideEffects": 0, "mayLoad": 0, "mayStore": 0}
    match mnemonic:
        case "flh" | "flah" | "flb" | "flab":
            properties["mayLoad"] = 1
        case "fsh" | "fsah" | "fsb" | "fsab":
            properties["mayStore"] = 1
    return properties


def to_tablegen_def(inst: Instruction, extension: str) -> str:
    dtype = get_dtypes(inst.mnemonic)
    template = jinja2.Template(TBLGEN_TEMPLATES[inst.format])
    tblgendef = extension_to_defprefix(extension) + inst.mnemonic.upper().replace(
        ".", "_"
    )
    properties = get_properties(inst.mnemonic)
    properties["DecoderNamespace"] = '"{}"'.format(extension_to_decoderns(extension))
    args = {
        "def": tblgendef,
        "mnemonic": inst.mnemonic.replace("_", "."),
        "dtype": dtype,
        "properties": properties,
        # Add all known encoding fields:
        **asdict(inst.encoding),
    }
    return template.render(**args)


def to_tablegen_alias(
    inst: Instruction,
    extension: str,
    uses_mnemonic: str,
    uses_extension: str,
    defprefix: str | None = None,
) -> str:
    dtype = get_dtypes(inst.mnemonic)

    template = jinja2.Template(TBLGEN_ALIAS_TEMPLATES[inst.format])
    use = uses_mnemonic.upper().replace(".", "_")
    if defprefix:
        use = defprefix + use

    args = {"mnemonic": inst.mnemonic.replace("_", "."), "dtype": dtype, "use": use}

    # FIXME upstream FCVT only needs frm
    if (
        inst.format == InstructionFormat.I
        and uses_extension != "rv_xsflts"
        and uses_mnemonic.startswith("fcvt")
    ):
        args["rm"] = "FRM_DYN"

    return template.render(**args)


@dataclass
class EncodingCheck:
    asm: str
    encoding: str


def to_lit_test(instructions: dict[str, Instruction], extension: str) -> str:
    asm_operands = {
        InstructionFormat.R: ("ft0", "ft0", "ft0"),
        InstructionFormat.I: ("ft0", "ft0"),
        InstructionFormat.S: ("ft0", "ft0"),
        InstructionFormat.U: ("ft0"),
        InstructionFormat.R4: ("ft0", "ft0", "ft0", "ft0"),
        InstructionFormat.RVF: ("ft0", "ft0", "ft0"),
        InstructionFormat.RFRM: ("ft0", "ft0", "ft0", "dyn"),
        InstructionFormat.IFRM: ("ft0", "ft0", "dyn"),
        InstructionFormat.R4FRM: ("ft0", "ft0", "ft0", "ft0", "dyn"),
        InstructionFormat.IIMM12: ("ft0", "ft0", "imm12"),
        InstructionFormat.SIMM12: ("ft0", "ft0", "imm12lo", "imm12hi"),
        InstructionFormat.IVF: ("ft0", "ft0"),
    }
    checks = []
    for _, inst in instructions.items():
        print(inst)
        operands = asm_operands[inst.format]
        asm = "{} {}".format(inst.mnemonic, ", ".join(operands))
        # Expected encoding:
        encoding = inst.encoding_repr
        encoding = encoding.replace("-", "0")
        assert len(encoding) == 32
        chunks_repr = re.findall(".{8}", encoding)
        assert len(chunks_repr) == 4
        chunks_int = (int(v, 2) for v in chunks_repr)
        chunks_hex = ("0x{:02x}".format(v) for v in chunks_int)
        chunks = "[{}]".format(",".join(chunks_hex))
        #
        checks.append(EncodingCheck(asm, chunks))
    template = jinja2.Template(LIT_FILE_TEMPLATE)
    return template.render(ext=extension, checks=checks)


def main():
    parser = argparse.ArgumentParser(
        description="Process an instruction YAML dictionary produced "
        "by riscv-opcodes and emits tablegen instruction definitions for the LLVM backend."
    )
    parser.add_argument(
        "--ext",
        type=str,
        required=True,
        help="RISC-V extension name (e.g.: Xfoo)",
    )
    parser.add_argument(
        "--emit-test",
        type=str,
        help="Emit lit encoding roundtrip tests to the specified file.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Path to the input file. If not specified or '-', reads from stdin.",
    )
    args = parser.parse_args()
    log.getLogger().setLevel(log.INFO)

    if args.input == "-":
        input = yaml.safe_load(sys.stdin)
    else:
        with open(args.input, "r") as file:
            input = yaml.safe_load(file)

    print("// Auto-generated by:")
    print("// $ {}".format(" ".join(sys.argv)))

    instructions: dict[str, Instruction] = {}
    pseudos: List[Instruction] = []
    uses: List[Tuple[str, str]] = []

    extension: str = args.ext

    for mnemonic, spec in input.items():
        mnemonic: str = mnemonic.replace("_", ".")
        inst = Instruction.from_dict(mnemonic, spec)
        if "is_pseudo_of" in spec:
            inst_use = spec["is_pseudo_of"]["instruction"]
            ext_use = spec["is_pseudo_of"]["extension"]
            pseudos.append(inst)
            uses.append((inst_use, ext_use))
        else:
            instructions[mnemonic] = inst

    # Instruction definitions

    for _, inst in instructions.items():
        print(to_tablegen_def(inst, extension))

    # Instruction aliases

    for pseudo, (inst_use, ext_use) in zip(pseudos, uses):
        defprefix = None
        if inst_use in instructions:
            defprefix = extension_to_defprefix(extension)
        else:
            log.info(
                f"Alias to another extension: {pseudo.mnemonic:10} -> {ext_use}::{inst_use}"
            )
        print(to_tablegen_alias(pseudo, extension, inst_use, ext_use, defprefix))

    # Encoding roundtrip tests

    if args.emit_test:
        with open(args.emit_test, "w") as f:
            f.write(to_lit_test(instructions, extension))


if __name__ == "__main__":
    main()
