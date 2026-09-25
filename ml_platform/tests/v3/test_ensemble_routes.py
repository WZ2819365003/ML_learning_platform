"""Route wiring for ensemble deployments.

`POST /api/deploy/{task_id}` is a single-segment path parameter, so it swallows
every literal sibling registered after it. The ensemble endpoints were appended
to the end of the module and every 创建融合部署 click landed on the
single-model handler with task_id="ensembles", which 404'd looking for a model
by that name — an error that says nothing about the real cause.

These assert the route *identity*, not just that some route exists, because a
404 from the wrong handler looks identical to a missing route.
"""
from __future__ import annotations

from app.main import create_app


def _route_for(app, method: str, path: str):
    for route in app.routes:
        if getattr(route, "path", None) == path and method in getattr(route, "methods", set()):
            return route
    return None


class TestEnsembleRouteOrdering:
    def test_post_ensembles_resolves_to_the_ensemble_handler(self):
        app = create_app()
        route = _route_for(app, "POST", "/api/deploy/ensembles")
        assert route is not None, "POST /api/deploy/ensembles is not registered at all"
        assert route.endpoint.__name__ == "create_ensemble_route"

    def test_ensembles_is_declared_before_the_task_id_catch_all(self):
        """The ordering itself, so moving the block back down fails here."""
        app = create_app()
        paths = [getattr(r, "path", "") for r in app.routes]
        ensembles = paths.index("/api/deploy/ensembles")
        catch_all = paths.index("/api/deploy/{task_id}")
        assert ensembles < catch_all, (
            "/api/deploy/ensembles must be registered before /api/deploy/{task_id}, "
            "otherwise the catch-all matches first and the endpoint is unreachable"
        )

    def test_single_model_deploy_route_still_exists(self):
        app = create_app()
        route = _route_for(app, "POST", "/api/deploy/{task_id}")
        assert route is not None
        assert route.endpoint.__name__ == "deploy_model"

    def test_ensemble_predict_is_registered_outside_the_api_prefix(self):
        # Inference lives at the root, matching the single-model predict route.
        app = create_app()
        route = _route_for(app, "POST", "/inference/ensembles/{ensemble_id}/predict")
        assert route is not None
        assert route.endpoint.__name__ == "ensemble_predict_route"
