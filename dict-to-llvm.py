#!/usr/bin/env python3

import re
import yaml
import sys
import argparse
import jinja2
from typing import Set, Self
from enum import Enum, IntEnum, auto, unique
from dataclasses import dataclass, asdict


def extract_bits(value: int, offset: int, n: int) -> str:
    decimal = (value >> offset) & ((1 << n) - 1)
    bin = "{0:b}".format(decimal)
    return "0b" + bin.zfill(n)


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
    # Xsflt
    rm: str

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
            vecfltop=extract_bits(encoding, 25, 4),
            r=extract_bits(encoding, 14, 1),
            vfmt=extract_bits(encoding, 12, 2),
            # Xsflt
            rm=extract_bits(encoding, 12, 2),
        )

    @classmethod
    def from_string(cls, encoding: str) -> Self:
        return cls.from_int(int(encoding, 0))


@dataclass
class Instruction:

    mnemonic: str
    encoding: Encoding
    format: InstructionFormat

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
        )


TBLGEN_TEMPLATE_R = """
def {{def}} : RVInstR<{
                {{funct7}}, // funct7
                {{funct3}}, // funct3
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2),
                "{{mnemonic}}", "$rd, $rs1, $rs2">,
                Sched<[]>;
"""

TBLGEN_TEMPLATE_IIMM12 = """
def {{def}} : RVInstI<
                {{funct7}}, // funct7
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, simm12:$imm12),
                "{{mnemonic}}", "$rd, $rs1, ${imm12}(${rs1})">,
                Sched<[]>;
"""

TBLGEN_TEMPLATE_SIMM12 = """
def {{def}} : RVInstS<
                {{funct7}}, // funct7
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs ),
                (ins {{dtype["rs2"]}}:$rs2, {{dtype["rs1"]}}:$rs1, simm12:$imm12),
                "{{mnemonic}}", "$rs2, ${imm12}(${rs1})">,
                Sched<[]>;
"""

TBLGEN_TEMPLATE_R4 = """
def {{def}} : RVInstR4<
                {{funct2}}, // funct2
                {{funct3}}, // funct3
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rs3"]}}:$rs3),
                "{{mnemonic}}", "$rd, $rs1, $rs2, $rs3">,
                Sched<[]>;
"""

TBLGEN_TEMPLATE_R4FRM = """
def {{def}} : RVInstR4Frm<
                {{funct7}}, // funct7
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rs3"]}}:$rs3, frmarg:$frm),
                "{{mnemonic}}", "$rd, $rs1, $rs2, $rs3, $frm">,
                Sched<[]>;
def: InstAlias<"{{mnemonic}} $rd, $rs1, $rs2, $rs3",
               ({{def}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, {{dtype["rs2"]}}:$rs2, {{dtype["rd"]}}:$rs3, FRM_DYN)>;
"""

TBLGEN_TEMPLATE_RFRM = """
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

TBLGEN_TEMPLATE_RVF = """
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

TBLGEN_TEMPLATE_IVF = """
def {{def}} : RVInstRVf<
                {{f2}}, // f2
                {{vecfltop}}, // vecfltop
                {{r}}, // r
                {{vfmt}}, // vfmt
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1,
                "{{mnemonic}}", "$rd, $rs1">,
                Sched<[]>
                { let rs2 = {{rs2}}; }
"""

TBLGEN_TEMPLATE_IFRM = """
def {{def}} : RVInstRFrm<
                {{funct7}}, // funct7
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1, frmarg:$frm),
                "{{mnemonic}}", "$rd, $rs1, $frm">,
                Sched<[]>
                { let rs2 = {{rs2}}; }
def: InstAlias<"{{mnemonic}} $rd, $rs1, $rs2",
               ({{def}} {{dtype["rd"]}}:$rd, {{dtype["rs1"]}}:$rs1, FRM_DYN)>;
"""

TBLGEN_TEMPLATE_I = """
def {{def}} : RVInstR<
                {{funct7}}, // funct7
                {{funct3}}, // funct3
                RISCVOpcode<"{{def}}", {{opcode}}>,
                (outs {{dtype["rd"]}}:$rd),
                (ins {{dtype["rs1"]}}:$rs1),
                "{{mnemonic}}", "$rd, $rs1">,
                Sched<[]>
                { let rs2 = {{rs2}}; }
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


def parse_dtypes(mnemonic: str) -> dict[str, str]:
    regs = ("rs1", "rs2", "rs3", "rd")
    match mnemonic:
        case "flh" | "fsh":
            return {r: TBLGEN_OPERAND_TYPES[DataType.f16] for r in regs}
        case "flah" | "fsah":
            return {r: TBLGEN_OPERAND_TYPES[DataType.f16alt] for r in regs}
        case "flb" | "fsb" | "flab" | "fsab":
            return {r: TBLGEN_OPERAND_TYPES[DataType.f8] for r in regs}

    # Vector format with R: it's just noise in the instruction naming, let's remove it
    if is_vector(mnemonic):
        mnemonic = re.sub("[\\._][rR]", "", mnemonic)

    # Reasonable mnemonics
    inst_t = mnemonic.upper().split("_")[1:]
    # fcvt.ah.s <- source
    #      ^
    #      dest
    source_t = TBLGEN_OPERAND_TYPES[DataType.from_str(inst_t[-1])]
    dest_t = TBLGEN_OPERAND_TYPES[DataType.from_str(inst_t[0])]
    return {
        "rs1": source_t,
        "rs2": source_t,
        "rs3": source_t,
        "rd": dest_t,
    }


def to_tablegen(inst: Instruction, defprefix=None) -> str:
    dtype = parse_dtypes(inst.mnemonic)
    template = jinja2.Template(TBLGEN_TEMPLATES[inst.format])
    tblgendef = inst.mnemonic.upper().replace(".", "_")
    if defprefix:
        tblgendef = defprefix + tblgendef
    args = {
        "def": tblgendef,
        "mnemonic": inst.mnemonic.replace("_", "."),
        "dtype": dtype,
        # Add all known encoding fields:
        **asdict(inst.encoding),
    }
    return template.render(**args)


def main():
    parser = argparse.ArgumentParser(
        description="Process an instruction YAML dictionary produced "
        "by riscv-opcodes and emits tablegen instruction definitions for the LLVM backend."
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default=None,
        help="Prefix for Tablegen definitions.",
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
        if "is_pseudo_of" in spec:
            continue
        inst = Instruction.from_dict(mnemonic, spec)
        # print("// {}".format(inst))
        print(to_tablegen(inst, args.prefix))


if __name__ == "__main__":
    main()
