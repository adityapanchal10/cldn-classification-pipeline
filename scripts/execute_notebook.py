import papermill as pm


def execute_notebook(input_path, output_path, parameters):

    pm.execute_notebook(
        input_path=input_path,
        output_path=output_path,
        parameters=parameters,
        kernel_name="python3",
    )
