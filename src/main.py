from src.pipeline import SarcasmDetectionPipeline

if __name__ == '__main__':
    try:
        pipeline = SarcasmDetectionPipeline()
        pipeline.run_pipeline()
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado - {e}")
    except ValueError as e:
        print(f"Error de valor: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
