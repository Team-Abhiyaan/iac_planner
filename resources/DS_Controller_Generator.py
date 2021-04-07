#! /usr/bin/python3

def replace(template: str, output_file: str, replacements: dict):
    with open(template) as t_file, open(output_file, 'w') as o_file:
        for line in t_file:
            for field, value in replacements.items():
                line = line.replace(field, value)
            o_file.write(line)


def main():
    template = 'resources/RtiSCADE_DS_Controller.xml.template'
    output_pattern = 'resources/RtiSCADE_DS_Controller_ego%d.xml'

    print(f"Using {template}")
    for i in range(1, 8 + 1):
        print(f"Creating {output_pattern % i}")
        replace(template, output_pattern % i, {"&ego_id;": str(i)})


if __name__ == '__main__':
    main()
