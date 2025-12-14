import os
import io
import json
from datetime import datetime
from flask import session, render_template, request, jsonify, send_file, redirect, url_for
from flask_login import current_user
import pandas as pd

from app import app, db
from replit_auth import require_login, make_replit_blueprint
from models import SolvedProblem
from quantum_solver import get_equations, solve_equation, generate_graph

app.register_blueprint(make_replit_blueprint(), url_prefix="/auth")


@app.before_request
def make_session_permanent():
    session.permanent = True
    if 'session_runs' not in session:
        session['session_runs'] = []


@app.route('/')
def index():
    return render_template('index.html', user=current_user if current_user.is_authenticated else None)


@app.route('/library')
def library():
    equations = get_equations()
    return render_template('library.html', 
                          equations=equations, 
                          user=current_user if current_user.is_authenticated else None)


@app.route('/solver/<equation_type>')
def solver(equation_type):
    equations = get_equations()
    if equation_type not in equations:
        return redirect(url_for('library'))
    equation = equations[equation_type]
    return render_template('solver.html', 
                          equation_type=equation_type, 
                          equation=equation,
                          user=current_user if current_user.is_authenticated else None)


@app.route('/api/solve', methods=['POST'])
def api_solve():
    data = request.json
    equation_type = data.get('equation_type')
    params = data.get('parameters', {})
    
    for key, value in params.items():
        try:
            if '.' in str(value):
                params[key] = float(value)
            else:
                params[key] = int(value)
        except (ValueError, TypeError):
            params[key] = float(value) if value else 0
    
    solution, steps, graph_data = solve_equation(equation_type, params)
    
    if 'error' in solution:
        return jsonify({'error': solution['error']}), 400
    
    graph_base64 = generate_graph(graph_data) if graph_data else None
    
    equations = get_equations()
    run_data = {
        'id': len(session.get('session_runs', [])) + 1,
        'equation_type': equation_type,
        'equation_name': equations[equation_type]['name'],
        'parameters': params,
        'solution': solution,
        'steps': steps,
        'graph_data': graph_data,
        'graph_base64': graph_base64,
        'timestamp': datetime.now().isoformat()
    }
    
    if 'session_runs' not in session:
        session['session_runs'] = []
    session['session_runs'].append(run_data)
    session.modified = True
    
    if current_user.is_authenticated:
        problem = SolvedProblem(
            user_id=current_user.id,
            equation_type=equation_type,
            equation_name=equations[equation_type]['name'],
            parameters=json.dumps(params),
            solution=json.dumps(solution),
            steps=json.dumps(steps),
            graph_data=json.dumps(graph_data) if graph_data else None
        )
        db.session.add(problem)
        db.session.commit()
        run_data['saved_id'] = problem.id
    
    return jsonify({
        'success': True,
        'solution': solution,
        'steps': steps,
        'graph': graph_base64,
        'run_id': run_data['id']
    })


@app.route('/problems')
def problems():
    user_problems = []
    if current_user.is_authenticated:
        user_problems = SolvedProblem.query.filter_by(user_id=current_user.id).order_by(SolvedProblem.created_at.desc()).all()
    return render_template('problems.html', 
                          problems=user_problems,
                          user=current_user if current_user.is_authenticated else None)


@app.route('/runs')
def runs():
    session_runs = session.get('session_runs', [])
    saved_problems = []
    if current_user.is_authenticated:
        saved_problems = SolvedProblem.query.filter_by(user_id=current_user.id).order_by(SolvedProblem.created_at.desc()).all()
    return render_template('runs.html', 
                          runs=session_runs,
                          saved_problems=saved_problems,
                          user=current_user if current_user.is_authenticated else None)


@app.route('/account')
@require_login
def account():
    user_problems = SolvedProblem.query.filter_by(user_id=current_user.id).order_by(SolvedProblem.created_at.desc()).all()
    return render_template('account.html', 
                          user=current_user,
                          problems=user_problems)


@app.route('/api/problem/<int:problem_id>')
def get_problem(problem_id):
    if current_user.is_authenticated:
        problem = SolvedProblem.query.filter_by(id=problem_id, user_id=current_user.id).first()
        if problem:
            graph_base64 = None
            if problem.graph_data:
                graph_data = json.loads(problem.graph_data)
                graph_base64 = generate_graph(graph_data)
            
            return jsonify({
                'id': problem.id,
                'equation_type': problem.equation_type,
                'equation_name': problem.equation_name,
                'parameters': json.loads(problem.parameters),
                'solution': json.loads(problem.solution),
                'steps': json.loads(problem.steps),
                'graph': graph_base64,
                'created_at': problem.created_at.isoformat()
            })
    return jsonify({'error': 'Problem not found'}), 404


@app.route('/api/run/<int:run_id>')
def get_run(run_id):
    session_runs = session.get('session_runs', [])
    for run in session_runs:
        if run['id'] == run_id:
            return jsonify(run)
    return jsonify({'error': 'Run not found'}), 404


@app.route('/export/graph/<int:problem_id>')
def export_graph(problem_id):
    if current_user.is_authenticated:
        problem = SolvedProblem.query.filter_by(id=problem_id, user_id=current_user.id).first()
        if problem and problem.graph_data:
            graph_data = json.loads(problem.graph_data)
            
            buf = io.BytesIO()
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            fig.patch.set_facecolor('#1a1a2e')
            
            ax1.set_facecolor('#16213e')
            ax1.plot(graph_data['x'], graph_data['psi'], color='#00d9ff', linewidth=2)
            ax1.set_xlabel(graph_data['x_label'], color='#e0e0e0')
            ax1.set_ylabel('Wavefunction ψ(x)', color='#e0e0e0')
            ax1.set_title(graph_data['title'], color='#ffffff', fontsize=14, fontweight='bold')
            ax1.tick_params(colors='#e0e0e0')
            ax1.grid(True, alpha=0.3, color='#4a4a6a')
            
            ax2.set_facecolor('#16213e')
            ax2.fill_between(graph_data['x'], graph_data['prob'], color='#ff6b9d', alpha=0.5)
            ax2.plot(graph_data['x'], graph_data['prob'], color='#ff6b9d', linewidth=2)
            ax2.set_xlabel(graph_data['x_label'], color='#e0e0e0')
            ax2.set_ylabel('Probability Density |ψ|²', color='#e0e0e0')
            ax2.tick_params(colors='#e0e0e0')
            ax2.grid(True, alpha=0.3, color='#4a4a6a')
            
            plt.tight_layout()
            plt.savefig(buf, format='jpeg', dpi=150, facecolor='#1a1a2e')
            buf.seek(0)
            plt.close()
            
            return send_file(buf, mimetype='image/jpeg', as_attachment=True, 
                           download_name=f'quantum_graph_{problem_id}.jpg')
    
    return jsonify({'error': 'Not found'}), 404


@app.route('/export/excel/<int:problem_id>')
def export_excel(problem_id):
    if current_user.is_authenticated:
        problem = SolvedProblem.query.filter_by(id=problem_id, user_id=current_user.id).first()
        if problem:
            params = json.loads(problem.parameters)
            solution = json.loads(problem.solution)
            steps = json.loads(problem.steps)
            
            buf = io.BytesIO()
            
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                info_df = pd.DataFrame({
                    'Property': ['Equation Type', 'Equation Name', 'Solved On'],
                    'Value': [problem.equation_type, problem.equation_name, problem.created_at.strftime('%Y-%m-%d %H:%M:%S')]
                })
                info_df.to_excel(writer, sheet_name='Info', index=False)
                
                params_df = pd.DataFrame(list(params.items()), columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
                
                solution_df = pd.DataFrame(list(solution.items()), columns=['Result', 'Value'])
                solution_df.to_excel(writer, sheet_name='Solution', index=False)
                
                steps_df = pd.DataFrame({'Step': steps})
                steps_df.to_excel(writer, sheet_name='Steps', index=False)
                
                if problem.graph_data:
                    graph_data = json.loads(problem.graph_data)
                    data_df = pd.DataFrame({
                        'x': graph_data['x'],
                        'psi': graph_data['psi'],
                        'probability': graph_data['prob']
                    })
                    data_df.to_excel(writer, sheet_name='Graph Data', index=False)
            
            buf.seek(0)
            return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           as_attachment=True, download_name=f'quantum_solution_{problem_id}.xlsx')
    
    return jsonify({'error': 'Not found'}), 404


@app.route('/export/session/graph/<int:run_id>')
def export_session_graph(run_id):
    session_runs = session.get('session_runs', [])
    for run in session_runs:
        if run['id'] == run_id and run.get('graph_data'):
            graph_data = run['graph_data']
            
            buf = io.BytesIO()
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.style.use('dark_background')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            fig.patch.set_facecolor('#1a1a2e')
            
            ax1.set_facecolor('#16213e')
            ax1.plot(graph_data['x'], graph_data['psi'], color='#00d9ff', linewidth=2)
            ax1.set_xlabel(graph_data['x_label'], color='#e0e0e0')
            ax1.set_ylabel('Wavefunction ψ(x)', color='#e0e0e0')
            ax1.set_title(graph_data['title'], color='#ffffff', fontsize=14, fontweight='bold')
            ax1.tick_params(colors='#e0e0e0')
            ax1.grid(True, alpha=0.3, color='#4a4a6a')
            
            ax2.set_facecolor('#16213e')
            ax2.fill_between(graph_data['x'], graph_data['prob'], color='#ff6b9d', alpha=0.5)
            ax2.plot(graph_data['x'], graph_data['prob'], color='#ff6b9d', linewidth=2)
            ax2.set_xlabel(graph_data['x_label'], color='#e0e0e0')
            ax2.set_ylabel('Probability Density |ψ|²', color='#e0e0e0')
            ax2.tick_params(colors='#e0e0e0')
            ax2.grid(True, alpha=0.3, color='#4a4a6a')
            
            plt.tight_layout()
            plt.savefig(buf, format='jpeg', dpi=150, facecolor='#1a1a2e')
            buf.seek(0)
            plt.close()
            
            return send_file(buf, mimetype='image/jpeg', as_attachment=True,
                           download_name=f'quantum_graph_session_{run_id}.jpg')
    
    return jsonify({'error': 'Not found'}), 404


@app.route('/export/session/excel/<int:run_id>')
def export_session_excel(run_id):
    session_runs = session.get('session_runs', [])
    for run in session_runs:
        if run['id'] == run_id:
            buf = io.BytesIO()
            
            with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                info_df = pd.DataFrame({
                    'Property': ['Equation Type', 'Equation Name', 'Solved On'],
                    'Value': [run['equation_type'], run['equation_name'], run['timestamp']]
                })
                info_df.to_excel(writer, sheet_name='Info', index=False)
                
                params_df = pd.DataFrame(list(run['parameters'].items()), columns=['Parameter', 'Value'])
                params_df.to_excel(writer, sheet_name='Parameters', index=False)
                
                solution_df = pd.DataFrame(list(run['solution'].items()), columns=['Result', 'Value'])
                solution_df.to_excel(writer, sheet_name='Solution', index=False)
                
                steps_df = pd.DataFrame({'Step': run['steps']})
                steps_df.to_excel(writer, sheet_name='Steps', index=False)
                
                if run.get('graph_data'):
                    data_df = pd.DataFrame({
                        'x': run['graph_data']['x'],
                        'psi': run['graph_data']['psi'],
                        'probability': run['graph_data']['prob']
                    })
                    data_df.to_excel(writer, sheet_name='Graph Data', index=False)
            
            buf.seek(0)
            return send_file(buf, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                           as_attachment=True, download_name=f'quantum_solution_session_{run_id}.xlsx')
    
    return jsonify({'error': 'Not found'}), 404


@app.route('/api/delete_problem/<int:problem_id>', methods=['DELETE'])
@require_login
def delete_problem(problem_id):
    problem = SolvedProblem.query.filter_by(id=problem_id, user_id=current_user.id).first()
    if problem:
        db.session.delete(problem)
        db.session.commit()
        return jsonify({'success': True})
    return jsonify({'error': 'Problem not found'}), 404


@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403


@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404
