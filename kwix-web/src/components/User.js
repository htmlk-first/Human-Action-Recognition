import { observer } from "mobx-react";
import React from "react";
import { user } from "../services/api";
import { useNavigate } from "react-router-dom";

export const User = observer(({ auth }) => {
	const navigate = useNavigate();

	const [userInfo, setUserInfo] = React.useState({
		email: "",
		name: "",
		id: "",
		height: 0,
		weight: 0,
		sex: 0,
		age: 0,
		bmi: 0,
		during: 0,
	});
	React.useEffect(() => {
		if (auth.isAuth) {
			const email = localStorage.getItem("kwixUser");
			user.get(email)
				.then((res) => {
					setUserInfo(res.data);
				})
				.catch((res) => {
					console.error(res);
				});
		}
	}, []);

	return (
		<div>
			<div className="container py-5 h-100">
				<div className="row justify-content-center align-items-center h-100">
					<div className="col-12 col-lg-9 col-xl-7">
						<div
							className="card shadow-2-strong card-registration"
							style={{ borderRadius: "15px" }}
						>
							<div className="card-body p-4 p-md-5">
								<h3 className="mb-4 pb-2 pb-md-0 mb-md-5">
									내 정보
								</h3>
								<form>
									<div className="row">
										<div className="col-md-6 mb-4">
											<div className="card2">
												<div className="card-header">
													이름
												</div>
												<div className="card-body">
													{userInfo.name}
												</div>
											</div>
										</div>
									</div>

									<div className="row">
										<div className="col-md-6 mb-4">
											<div className="card2">
												<div className="card-header">
													성별
												</div>
												<div className="card-body">
													{userInfo.sex}
												</div>
											</div>
										</div>
									</div>

									<div className="row">
										<div className="col-md-6 mb-4">
											<div className="card2">
												<div className="card-header">
													나이
												</div>
												<div className="card-body">
													{userInfo.age}
												</div>
											</div>
										</div>
									</div>

									<div className="row">
										<div className="col-md-6 mb-4">
											<div className="card2">
												<div className="card-header">
													키
												</div>
												<div className="card-body">
													{userInfo.height}
												</div>
											</div>
										</div>
									</div>

									<div className="row">
										<div className="col-md-6 mb-4">
											<div className="card2">
												<div className="card-header">
													몸무게
												</div>
												<div className="card-body">
													{userInfo.weight}
												</div>
											</div>
										</div>
									</div>

									<div className="row">
										<div className="col-md-6 mb-4">
											<div className="card2">
												<div className="card-header">
													BMI
												</div>
												<div className="card-body">
													{userInfo.bmi}
												</div>
											</div>
										</div>
									</div>

									<div className="row">
										<div className="col-md-6 mb-4">
											<div className="card2">
												<div className="card-header">
													운동 경력
												</div>
												<div className="card-body">
													{userInfo.during}
												</div>
											</div>
										</div>
									</div>

									<div className="mt-4 pt-2">
										<button
											className="w-50 btn btn-lg btn-primary"
											type="button"
											onClick={() => navigate("/input")}
										>
											내정보 수정
										</button>
									</div>
								</form>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>
	);
});
