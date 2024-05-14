#!/usr/bin/env python3

import yaml
import sys
import argparse
from enum import Enum, IntEnum, auto, unique
from dataclasses import dataclass, asdict
from typing import Set
import jinja2
import re


def to_bin(value: int) -> str:
    return "0b{0:b}".format(value)


def extract_bits(value: int, offset: int, n: int) -> int:
    return (value >> offset) & ((1 << n) - 1)


def is_vector(mnemonic: str) -> bool:
    return mnemonic.startswith(("V", "v"))


@unique
class DataType(Enum):
    f8 = "B"
    f8alt = "AB"  # bfloat8?
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
    def from_str(cls, type: str):
        for label in cls:
            if type == label.value:
                return label
        raise ValueError(f"DataType: cannot recognize data type from string: '{type}'")


@unique
class InstructionFormat(IntEnum):
    R = 0
    I = auto()
    S = auto()
    U = auto()
    R4 = auto()
    RVF = auto()
    # Variants of standard formats that rename a known encoding
    # field to match its semantics (e.g.: funct3 -> rm):
    RFRM = auto()
    IFRM = auto()
    R4FRM = auto()
    IIMM12 = auto()
    SIMM12 = auto()
    IVF = auto()

    def operands(self) -> Set[str]:
        return self.__cls__.format_operands()[self]

    @classmethod
    def format_operands(cls):
        return {
            cls.R: {"rd", "rs1", "rs2"},
            cls.I: {"rd", "rs1"},
            cls.S: {"rs1", "rs2"},
            cls.U: {"rd"},
            cls.R4: {"rd", "rs1", "rs2", "rs3"},
            cls.RVF: {"rs2", "rs1", "rd"},
            # Variants of standard formats that rename a known encoding
            # field to match its semantics (e.g.: funct3 -> rm):
            cls.RFRM: {"rd", "rs1", "rs2", "rm"},
            cls.IFRM: {"rd", "rs1", "rm"},
            cls.R4FRM: {"rs1", "rs2", "rs3", "rd", "rm"},
            cls.IIMM12: {"rs1", "rd", "imm12"},
            cls.SIMM12: {"rs1", "rs2", "imm12lo", "imm12hi"},
            cls.IVF: {"rs1", "rd"},
        }

    @classmethod
    def from_operands(cls, operands: Set[str]):
        for fmt, ops in cls.format_operands().items():
            if operands == ops:
                return fmt
        raise ValueError(
            f"InstructionFormat: cannot recognize instruction format from operands: {operands}"
        )


@dataclass
class Encoding:

    opcode: int
    funct3: int
    rs2: int
    csr: int
    funct7: int
    # Vector fields
    f2: int
    vecfltop: int
    r: int
    vfmt: int

    def to_bin(self) -> dict[str, str]:
        return {field: to_bin(value) for field, value in asdict(self).items()}

    @classmethod
    def from_int(cls, encoding: int):
        return cls(
            opcode=extract_bits(encoding, 0, 7),
            funct3=extract_bits(encoding, 12, 3),
            rs2=extract_bits(encoding, 20, 5),
            csr=extract_bits(encoding, 20, 12),
            funct7=extract_bits(encoding, 25, 7),
            # Vector fields
            f2=extract_bits(encoding, 30, 2),
            vecfltop=extract_bits(encoding, 25, 4),
            r=extract_bits(encoding, 14, 1),
            vfmt=extract_bits(encoding, 12, 2),
        )

    @classmethod
    def from_string(cls, encoding: str):
        return Encoding.from_int(int(encoding, 0))


@dataclass
class Instruction:

    mnemonic: str
    encoding: Encoding
    format: InstructionFormat

    @classmethod
    def from_dict(cls, mnemonic: str, spec: dict):
        fmt = InstructionFormat.from_operands(set(spec["variable_fields"]))
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
            encoding=Encoding.from_string(spec["match"]),
            format=fmt,
        )


TBLGEN_TEMPLATE_R = """
{% set def = mnemonic | upper | replace(".", "_") %}
def {{def}} : RVInstR<{{funct7}},
                      RISCVOpcode<"{{def}}", {{opcode}}>,
                      (outs {{dtype["rd"]}}:$rd),
                      (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2),
                      "{{mnemonic}}", "$rd, $rs1, $rs2">,
                      Sched<[]>;
"""

TBLGEN_TEMPLATE_IIMM12 = """
{% set def = mnemonic | upper | replace(".", "_") %}
def {{def}} : RVInstI<{{funct7}},
                      RISCVOpcode<"{{def}}", {{opcode}}>,
                      (outs {{dtype["rd"]}}:$rd),
                      (ins {{dtype["rs1"]}}:$rs1, simm12:$imm12),
                      "{{mnemonic}}", "$rd, $rs1, ${imm12}(${rs1})">,
                      Sched<[]>;
"""

TBLGEN_TEMPLATE_SIMM12 = """
{% set def = mnemonic | upper | replace(".", "_") %}
def {{def}} : RVInstS<{{funct7}},
                      RISCVOpcode<"{{def}}", {{opcode}}>,
                      (outs ),
                      (ins {{dtype["rs2"]}}:$rs2, {{dtype["rs1"]}}:$rs1, simm12:$imm12),
                      "{{mnemonic}}", "$rs2, ${imm12}(${rs1})">,
                      Sched<[]>;
"""

TBLGEN_TEMPLATE_R4FRM = """
{% set def = mnemonic | upper | replace(".", "_") %}
def {{def}} : RVInstR4Frm<{{funct7}},
                          RISCVOpcode<"{{def}}", {{opcode}}>,
                          (outs {{dtype["rd"]}}:$rd),
                          (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rs3"]}}:$rs3, frmarg:$frm),
                          "{{mnemonic}}", "$rd, $rs1, $rs2, $rs3, $frm">,
                          Sched<[]>;
def          : InstAlias<"{{mnemonic}} $rd, $rs1, $rs2, $rs3",
                         ({{def}} {{dtype["rd"]}}:$rd, dtype["rs1"]:$rs1, dtype["rs2"]:$rs2, {{dtype["rd"]}}:$rs3, FRM_DYN)>;
"""

TBLGEN_TEMPLATE_RFRM = """
{% set def = mnemonic | upper | replace(".", "_") %}
def {{def}} : RVInstRFrm<{{funct7}},
                         RISCVOpcode<"{{def}}", {{opcode}}>,
                         (outs {{dtype["rd"]}}:$rd),
                         (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, frmarg:$frm),
                         "{{mnemonic}}", "$rd, $rs1, $rs2, $frm">,
                         Sched<[]> ;
def      : InstAlias<"{{mnemonic}} $rd, $rs1, $rs2",
                     ({{def}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, FRM_DYN)>;
"""

TBLGEN_TEMPLATE_RVF = """
{% set def = mnemonic | upper | replace(".", "_") %}
def {{def}} : RVInstRVf<{{f2}}, // f2
                        {{vecfltop}}, // vecfltop
                        {{r}}, // r
                        {{vfmt}}, // vfmt
                        RISCVOpcode<"{{def}}", {{opcode}}>,
                        (outs {{dtype["rd"]}}:$rd),
                        (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2),
                        "{{mnemonic}}", "$rd, $rs1, $rs2">,
                        Sched<[]>;
"""

TBLGEN_TEMPLATES = {
    InstructionFormat.R: TBLGEN_TEMPLATE_R,
    # InstructionFormat.I: TBLGEN_TEMPLATE_I,
    # InstructionFormat.S: TBLGEN_TEMPLATE_S,
    # InstructionFormat.U: TBLGEN_TEMPLATE_U,
    # InstructionFormat.R4: TBLGEN_TEMPLATE_R4,
    InstructionFormat.RFRM: TBLGEN_TEMPLATE_RFRM,
    # InstructionFormat.IFRM: TBLGEN_TEMPLATE_IFRM,
    InstructionFormat.R4FRM: TBLGEN_TEMPLATE_R4FRM,
    InstructionFormat.IIMM12: TBLGEN_TEMPLATE_IIMM12,
    InstructionFormat.SIMM12: TBLGEN_TEMPLATE_SIMM12,
    InstructionFormat.RVF: TBLGEN_TEMPLATE_RVF,
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

# def parse_datatypes_from_mnemonic(mnemonic: str):
#     parts = mnemonic.upper().split('.')


def parse_dtypes(mnemonic: str) -> dict[str, str]:
    # Known instruction mnemonics that make no sense at all
    regs = ("rs1", "rs2", "rs3", "rd")
    match mnemonic:
        case "flah" | "fsah":
            return {r: TBLGEN_OPERAND_TYPES[DataType.f16alt] for r in regs}
        case "flb" | "fsb":
            return {r: TBLGEN_OPERAND_TYPES[DataType.f8] for r in regs}

    # Vector format with R: it's just noise in the instruction naming, let's remove it
    if is_vector(mnemonic):
        mnemonic = re.sub("[\._][rR]", "", mnemonic)

    # Reasonable mnemonics
    instruction_types = mnemonic.upper().split("_")[1:]
    return {
        "rs1": TBLGEN_OPERAND_TYPES[DataType.from_str(instruction_types[-1])],
        "rs2": TBLGEN_OPERAND_TYPES[DataType.from_str(instruction_types[-1])],
        "rs3": TBLGEN_OPERAND_TYPES[DataType.from_str(instruction_types[-1])],
        "rd": TBLGEN_OPERAND_TYPES[DataType.from_str(instruction_types[0])],
    }


def to_tablegen(inst: Instruction) -> str:
    dtype = parse_dtypes(inst.mnemonic.replace("@", ""))  # FIXME

    if inst.format not in TBLGEN_TEMPLATES:
        return "// TODO"

    template = jinja2.Template(TBLGEN_TEMPLATES[inst.format])

    args = {
        "mnemonic": inst.mnemonic.replace("_", ".").removeprefix("@"),
        "dtype": dtype,
        # Add all known encoding fields:
        **inst.encoding.to_bin(),
    }
    return template.render(**args)


def main():
    parser = argparse.ArgumentParser(
        description="Process input from positional argument or stdin"
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="-",
        help="Path to the input file. If not specified or '-', reads from stdin.",
    )
    args = parser.parse_args()

    if args.input == "-":
        input = yaml.safe_load(sys.stdin)
    else:
        with open(args.input, "r") as file:
            input = yaml.safe_load(file)

    for mnemonic, spec in input.items():
        inst = Instruction.from_dict(mnemonic, spec)
        print("// {}".format(inst))
        print(to_tablegen(inst))


if __name__ == "__main__":
    main()
