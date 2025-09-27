from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ..services.dataset_loader_service import load_o_shank_dataset
from .base_demo import BaseDemo


class SVRV3Demo(BaseDemo):
    def run(self):
        X, y = load_o_shank_dataset()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        C_values = (0.1, 1.0, 10.0)
        epsilon_values = (0.01, 0.1, 0.02)

        for C in C_values:
            for epsilon in epsilon_values:
                print("C={}, epsilon={}".format(C, epsilon))
                self.train_svr_model(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    C=C,
                    epsilon=epsilon,
                    kernel="rbf",
                )

        self.save_results("svr_v3.csv")
        self.show_results()
