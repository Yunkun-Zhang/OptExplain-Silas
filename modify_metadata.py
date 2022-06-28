import json
import os.path as osp
import argparse


def convert_nominal(
        metadata_file,
        metadata_settings: list = None,
        r: int = None,
        label_column: int = None,
        inplace: bool = True):
    """Convert nominal attributes to numerical attributes.

    Args:
        metadata_file: Path to metadata file.
        metadata_settings: None.
        r (int): Round number. If None, convert nominal attributes to
            numerical attributes.
        label_column (int): Label column. If None, use the last column.
        inplace (bool): If True, modify the metadata file inplace.
    """

    # load metadata
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    if metadata_settings is None:
        base_path = osp.dirname(metadata_file)
        metadata_settings_file = osp.join(base_path, 'metadata-settings.json')
        with open(metadata_settings_file, 'r') as f:
            settings = json.load(f)
    else:
        raise NotImplementedError

    # convert nominal to numerical
    label_column = label_column or len(metadata['attributes']) - 1
    for index, attr in enumerate(settings['attribute-settings']):
        # don't change label column
        if index == label_column:
            continue
        attribute = metadata['attributes'][index]
        if attr['type'] == 'numerical' and attribute['type'] == 'nominal':
            if r is None:
                attribute['type'] = 'numerical'
                attribute['data-type'] = 'f32'
                attribute['bounds'] = {
                    'min': min([float(v) for v in attribute['values']]),
                    'max': max([float(v) for v in attribute['values']])
                }
                del attribute['ordered']
                del attribute['values']
            else:
                assert r >= 0
                attribute['values'] = [
                    str(round(float(v), r)) for v in attribute['values']
                ]
        metadata['attributes'][index] = attribute

    if not inplace:
        metadata_file = metadata_file.replace('.json', '_new.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_file', type=str)
    parser.add_argument('-r', '--round_number', type=int, default=None)
    parser.add_argument('-l', '--label_column', type=int, default=None)
    parser.add_argument('--inplace', type=bool, default=True)
    args = parser.parse_args()
    convert_nominal(args.metadata_file, args.round_number, args.label_column, args.inplace)
